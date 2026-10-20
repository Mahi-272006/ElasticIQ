"""Leakage-safe feature engineering for demand forecasting.

Features are built on a long panel sorted by ``(id, day_idx)``. Every
lag or rolling feature at day ``t`` uses only values from days ``< t``:

- lag_k at day t = units at day t-k
- rolling statistics at day t are computed over days t-k..t-1
  (``shift(1)`` before the rolling window)
- calendar features (weekday, month, SNAP, events) are known in advance
- the current day's sell price is set before sales occur and is therefore
  usable at prediction time

The same builder is used for training rows and for recursive multi-step
inference, so training and serving features can never diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.pipeline import state_snap

LAG_DAYS = (1, 3, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28, 56)
STD_WINDOWS = (7, 28)

NUMERIC_FEATURES = (
    ["lag_1", "lag_3", "lag_7", "lag_14", "lag_28"]
    + [
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_28",
        "rolling_mean_56",
    ]
    + ["log1p_lag_1", "log1p_rolling_mean_28"]
    + ["sell_price", "price_rel_change_7"]
    + ["wday", "month", "is_event", "snap", "day_trend"]
)

CATEGORICAL_FEATURES = ["cat_id", "dept_id", "store_id", "state_id"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Recursive inference keeps this many recent days per series so that every
# rolling window (max 56 days) is fully covered inside the slice.
REQUIRED_HISTORY_DAYS = 60


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add per-series lag features (NaN when history is too short)."""

    grouped = frame.groupby("id", sort=False)["units"]
    for lag in LAG_DAYS:
        frame[f"lag_{lag}"] = grouped.shift(lag)
    return frame


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add per-series rolling mean/std features computed strictly on past days."""

    ids = frame["id"]
    shifted = frame.groupby("id", sort=False)["units"].shift(1)

    for window in ROLLING_WINDOWS:
        rolled = shifted.groupby(ids, sort=False).rolling(window, min_periods=1)
        frame[f"rolling_mean_{window}"] = rolled.mean().to_numpy()
        if window in STD_WINDOWS:
            frame[f"rolling_std_{window}"] = rolled.std(ddof=0).to_numpy()

    return frame


def add_transformed_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add log1p transforms and a linear time trend.

    Log features stabilize the scale difference between fast- and
    slow-moving series; the trend encodes slow drift and is known in
    advance for future days.
    """

    frame["log1p_lag_1"] = np.log1p(frame["lag_1"].astype(np.float32))
    frame["log1p_rolling_mean_28"] = np.log1p(
        frame["rolling_mean_28"].astype(np.float32)
    )
    frame["day_trend"] = (frame["day_idx"] / 1000.0).astype(np.float32)
    return frame


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add price history features relative to 7 days earlier."""

    prices = frame["sell_price"]
    week_ago = frame.groupby("id", sort=False)["sell_price"].shift(7)

    relative = (prices - week_ago) / week_ago.replace(0, np.nan)
    frame["price_rel_change_7"] = relative.fillna(0.0).astype(np.float32)

    # Price history: back-fill within a series so pre-introduction rows use
    # the first known price instead of NaN (price level, not change).
    frame["sell_price"] = frame.groupby("id", sort=False)["sell_price"].transform(
        lambda series: series.ffill()
    )
    return frame


def add_categorical_codes(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical feature columns to pandas 'category' dtype.

    HistGradientBoostingRegressor consumes these natively as categorical.
    """

    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("category")
    return frame


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add all model features to a long panel (id, day_idx sorted).

    The panel must provide columns: id, day_idx, units, sell_price, wday,
    month, is_event, snap, cat_id, dept_id, store_id, state_id.
    """

    frame = panel.sort_values(["id", "day_idx"], ignore_index=True).copy()

    frame = add_lag_features(frame)
    frame = add_rolling_features(frame)
    frame = add_price_features(frame)
    frame = add_transformed_features(frame)
    frame = add_categorical_codes(frame)

    for column in NUMERIC_FEATURES:
        if column not in ("wday", "month", "is_event", "snap"):
            frame[column] = frame[column].astype(np.float32)

    return frame


def build_inference_rows(
    history: pd.DataFrame,
    future_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature rows for future days of a single series.

    ``history`` is the recent panel rows of one series (id + metadata +
    calendar fields + units). ``future_calendar`` contains the calendar rows
    (cleaned) for the days to forecast; its ``sell_price`` column must hold
    the assumed future price (e.g. last known price).

    Rows for future days carry their true target (units) as NaN — the
    model must predict them.
    """

    history = history.sort_values("day_idx")
    last = history.iloc[-1]

    future = future_calendar.copy()
    future["id"] = last["id"]
    for column in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        future[column] = last[column]
    future["units"] = np.nan
    future["sell_price"] = future["sell_price"].fillna(last["sell_price"])

    snap_rows = future.reset_index(drop=True)
    future["snap"] = state_snap(snap_rows, future["state_id"]).to_numpy()

    combined = pd.concat(
        [
            history[
                [
                    "id",
                    "item_id",
                    "dept_id",
                    "cat_id",
                    "store_id",
                    "state_id",
                    "d",
                    "day_idx",
                    "units",
                    "sell_price",
                    "date",
                    "wm_yr_wk",
                    "wday",
                    "month",
                    "is_event",
                    "snap",
                ]
            ],
            future[
                [
                    "id",
                    "item_id",
                    "dept_id",
                    "cat_id",
                    "store_id",
                    "state_id",
                    "d",
                    "day_idx",
                    "units",
                    "sell_price",
                    "date",
                    "wm_yr_wk",
                    "wday",
                    "month",
                    "is_event",
                    "snap",
                ]
            ],
        ],
        ignore_index=True,
    )

    return build_features(combined)
