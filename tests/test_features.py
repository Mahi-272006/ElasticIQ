"""Tests for leakage-safe feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    EXTENDED_CATEGORICAL_FEATURES,
    EXTENDED_NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    build_features,
    build_inference_rows,
    categorical_columns,
    feature_columns,
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


def test_feature_column_selection_is_stable():
    """Base selection must not change (it is what the shipped model uses)."""

    assert feature_columns(extended=False) == FEATURE_COLUMNS
    assert categorical_columns(extended=False) == ["cat_id", "dept_id",
                                                   "store_id", "state_id"]

    extended = feature_columns(extended=True)
    # every base numeric feature except the two moved to categorical
    for column in FEATURE_COLUMNS:
        if column in ("wday", "month"):
            assert column not in [c for c in extended
                                  if c not in categorical_columns(True)]
            assert column in categorical_columns(True)
        else:
            assert column in extended
    for column in EXTENDED_NUMERIC_FEATURES:
        assert column in extended
    assert set(categorical_columns(True)) == set(categorical_columns(False)) | set(
        EXTENDED_CATEGORICAL_FEATURES
    )


def test_extended_features_exist_and_are_numeric():
    features = build_features(make_panel(n_days=70))
    for column in EXTENDED_NUMERIC_FEATURES:
        assert column in features.columns
        assert np.issubdtype(features[column].dtype, np.number), column


def test_no_future_information_leaks_into_extended_features():
    """Mutating future sales must not change earlier extended features."""

    panel = make_panel(n_days=70, series={"s1": lambda d: float(d % 5)})

    baseline_features = build_features(panel)
    mutated = panel.copy()
    mutated.loc[mutated["day_idx"] > 40, "units"] = 999.0
    mutated_features = build_features(mutated)

    early_base = baseline_features[baseline_features["day_idx"] <= 40]
    early_mutated = mutated_features[mutated_features["day_idx"] <= 40]

    pd.testing.assert_frame_equal(
        early_base[EXTENDED_NUMERIC_FEATURES].reset_index(drop=True),
        early_mutated[EXTENDED_NUMERIC_FEATURES].reset_index(drop=True),
    )


def test_intermittency_features_by_hand():
    """demand_share_56 and days_since_last_sale computed by hand."""

    # days 1..10 sell 1 unit every other day: sell on odd days only
    panel = make_panel(
        n_days=40,
        series={"s1": lambda d: 1.0 if d % 2 == 1 else 0.0},
    )
    features = build_features(panel)

    # At day 12 the past 10 days (2..11) are: 0,1,0,1,0,1,0,1,0,1 -> 5 sales
    # in the full past-10 window; past-56 window is just days 1..11 -> 6 sales
    # out of 11 observed days (day 1 has no past, shifted -> day 1 excluded).
    row12 = features[features["day_idx"] == 12].iloc[0]
    past = panel.loc[(panel["day_idx"] < 12) & (panel["day_idx"] >= 1), "units"]
    expected_share = float((past > 0).mean())
    assert row12["demand_share_56"] == pytest.approx(expected_share)

    # last sale before day 12 is day 11 -> 1 day since last sale
    assert row12["days_since_last_sale"] == 1.0

    # after a long quiet spell the feature saturates at 56
    quiet = make_panel(n_days=70, series={"q1": lambda d: 0.0})
    quiet_features = build_features(quiet)
    last = quiet_features[quiet_features["day_idx"] == 70].iloc[0]
    assert last["demand_share_56"] == 0.0
    assert last["days_since_last_sale"] == 56.0


def test_cv2_and_level_ratio_by_hand():
    # constant series: zero dispersion, momentum ratio exactly 1
    panel = make_panel(n_days=70, series={"s1": lambda d: 4.0})
    features = build_features(panel)
    row = features[features["day_idx"] == 40].iloc[0]
    assert row["cv2_28"] == pytest.approx(0.0, abs=1e-6)
    assert row["level_ratio_7_56"] == pytest.approx(1.0)


def test_extended_features_match_between_full_history_and_inference_tail():
    """Train/serve parity: the 60-day inference tail must reproduce exactly
    the same feature values as the full history for the forecast day."""

    series = {
        "s1": lambda d: float((d * 3) % 7),
        "s2": lambda d: 0.0 if d % 3 == 0 else 2.0,
        "s3": lambda d: float(d % 2),
    }
    panel = make_panel(n_days=90, series=series)

    full = build_features(panel).set_index(["id", "day_idx"])
    tail = build_features(panel[panel["day_idx"] > 30]).set_index(
        ["id", "day_idx"]
    )

    last_day = 90
    columns = feature_columns(extended=True)
    for series_id in series:
        full_row = full.loc[(series_id, last_day)]
        tail_row = tail.loc[(series_id, last_day)]
        for column in columns:
            left, right = full_row[column], tail_row[column]
            if isinstance(left, float) and np.isnan(left):
                assert np.isnan(right), f"{series_id}/{column} NaN mismatch"
            else:
                assert left == right, f"{series_id}/{column}: {left} != {right}"


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
