"""Tests for leakage-safe feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    FEATURE_COLUMNS,
    build_features,
    build_inference_rows,
)


def make_panel(
    n_days: int = 40,
    series: dict[str, callable] | None = None,
) -> pd.DataFrame:
    """Build a tiny synthetic long panel with all required columns."""

    series = series or {"s1": lambda d: (d % 7) + 2}
    rows = []

    for series_id, generator in series.items():
        for day in range(1, n_days + 1):
            rows.append(
                {
                    "id": series_id,
                    "item_id": "ITEM_1",
                    "dept_id": "DEPT_1",
                    "cat_id": "CAT_A",
                    "store_id": "STORE_1",
                    "state_id": "CA",
                    "d": f"d_{day}",
                    "day_idx": day,
                    "units": float(generator(day)),
                    "sell_price": 2.5,
                    "date": pd.Timestamp("2011-01-29") + pd.Timedelta(days=day - 1),
                    "wm_yr_wk": 11101 + (day - 1) // 7,
                    "wday": (day - 1) % 7 + 1,
                    "month": (
                        pd.Timestamp("2011-01-29") + pd.Timedelta(days=day - 1)
                    ).month,
                    "is_event": 0,
                    "snap": 0,
                    "snap_CA": 0,
                    "snap_TX": 0,
                    "snap_WI": 0,
                }
            )

    return pd.DataFrame(rows)


def test_lags_are_correct_by_hand():
    panel = make_panel(series={"s1": lambda d: float(d)})
    features = build_features(panel)

    row = features[features["day_idx"] == 10].iloc[0]
    assert row["lag_1"] == 9.0
    assert row["lag_7"] == 3.0
    assert np.isnan(row["lag_14"])

    row_30 = features[features["day_idx"] == 30].iloc[0]
    assert row_30["lag_28"] == 2.0


def test_rolling_mean_excludes_current_day():
    # Constant series: rolling mean of past days equals the constant.
    panel = make_panel(series={"s1": lambda d: 4.0})
    features = build_features(panel)

    row = features[features["day_idx"] == 15].iloc[0]
    assert row["rolling_mean_7"] == 4.0
    assert row["rolling_mean_28"] == 4.0
    assert row["rolling_std_7"] == 0.0


def test_rolling_mean_hand_computed():
    panel = make_panel(series={"s1": lambda d: float(d)})
    features = build_features(panel)

    # At day 10, rolling mean over days 3..9 = (3+4+5+6+7+8+9)/7 = 6.0
    row = features[features["day_idx"] == 10].iloc[0]
    assert row["rolling_mean_7"] == pytest.approx(6.0)


def test_no_future_information_leaks_into_features():
    """Mutating future sales must not change any earlier feature row."""

    panel = make_panel(n_days=50, series={"s1": lambda d: float(d % 5)})

    baseline_features = build_features(panel)

    mutated = panel.copy()
    mutated.loc[mutated["day_idx"] > 25, "units"] = 999.0
    mutated_features = build_features(mutated)

    columns = [c for c in FEATURE_COLUMNS if c not in ("cat_id",)]
    early_base = baseline_features[baseline_features["day_idx"] <= 25][columns]
    early_mutated = mutated_features[mutated_features["day_idx"] <= 25][columns]

    pd.testing.assert_frame_equal(early_base, early_mutated)


def test_categoricals_are_category_dtype():
    features = build_features(make_panel())
    assert str(features["cat_id"].dtype) == "category"
    assert str(features["store_id"].dtype) == "category"


def test_build_inference_rows_has_nan_targets_and_features():
    panel = make_panel(n_days=35)
    history = panel[panel["day_idx"] <= 30]

    # Future calendar rows carry every calendar column the builder needs;
    # their units are ignored (targets are set to NaN by the builder).
    future_calendar = panel[panel["day_idx"].between(31, 33)].copy()

    rows = build_inference_rows(history, future_calendar)

    future_rows = rows[rows["day_idx"] > 30]
    assert len(future_rows) == 3
    assert future_rows["units"].isna().all()

    # Features must be computable for the first future day.
    first_future = future_rows[future_rows["day_idx"] == 31].iloc[0]
    assert not pd.isna(first_future["lag_1"])
    assert not pd.isna(first_future["rolling_mean_7"])
    assert first_future["lag_1"] == history["units"].iloc[-1]
