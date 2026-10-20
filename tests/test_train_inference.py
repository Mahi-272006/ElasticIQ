"""Integration tests for training, artifacts, and recursive inference.

Uses a synthetic 70-day panel written directly into a temporary processed
directory, so no M5 data or long training runs are needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import clean_calendar
from src.features.engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    build_features,
)
from src.forecasting.inference import (
    forecast_from_history,
    latest_version,
    load_bundle,
    recursive_forecast,
)
from src.forecasting.train import chronological_split, run_training

N_DAYS = 70
CALENDAR_DAYS = 100  # calendar extends past sales, like the real M5 calendar
N_TRAIN_HEAD = 40  # days of history kept in the inference tail file


def synthetic_calendar(n_days: int = CALENDAR_DAYS) -> pd.DataFrame:
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")
    weekday_names = [
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ]
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": [11101 + i // 7 for i in range(n_days)],
            "weekday": [weekday_names[i % 7] for i in range(n_days)],
            "wday": [i % 7 + 1 for i in range(n_days)],
            "month": dates.month,
            "year": dates.year,
            "snap_CA": [1 if i % 3 == 0 else 0 for i in range(n_days)],
            "snap_TX": [1 if i % 4 == 0 else 0 for i in range(n_days)],
            "snap_WI": [1 if i % 5 == 0 else 0 for i in range(n_days)],
            "d": [f"d_{i + 1}" for i in range(n_days)],
        }
    )
    frame["event_name_1"] = pd.NA
    frame["event_type_1"] = pd.NA
    frame["event_name_2"] = pd.NA
    frame["event_type_2"] = pd.NA
    return frame


def synthetic_panel(calendar: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []

    for scale, (item, store, state, cat, dept) in enumerate(
        [
            ("ITEM_A", "STORE_1", "CA", "CAT_A", "DEPT_1"),
            ("ITEM_B", "STORE_1", "CA", "CAT_B", "DEPT_2"),
            ("ITEM_C", "STORE_2", "TX", "CAT_A", "DEPT_1"),
        ],
        start=1,
    ):
        series_id = f"{item}_{store}"
        base = 3.0 + 2.0 * scale
        for index, day in enumerate(calendar["d"]):
            weekly = 1.0 if index % 7 < 2 else 0.0
            units = base + weekly + rng.normal(0, 0.4)
            rows.append(
                {
                    "id": series_id,
                    "item_id": item,
                    "dept_id": dept,
                    "cat_id": cat,
                    "store_id": store,
                    "state_id": state,
                    "d": day,
                    "day_idx": index + 1,
                    "units": max(float(np.round(units)), 0.0),
                    "sell_price": 2.5 + 0.1 * scale,
                    "date": calendar["date"].iloc[index],
                    "wm_yr_wk": int(calendar["wm_yr_wk"].iloc[index]),
                    "wday": int(calendar["wday"].iloc[index]),
                    "month": int(calendar["month"].iloc[index]),
                    "is_event": 0,
                }
            )

    frame = pd.DataFrame(rows)
    # Attach snap like the pipeline does.
    cleaned = clean_calendar(calendar).set_index("d")
    snap_source = cleaned.loc[frame["d"]].reset_index(drop=True)
    from src.data.pipeline import state_snap

    frame["snap"] = state_snap(snap_source, frame["state_id"]).to_numpy()
    return frame


@pytest.fixture
def synthetic_processed(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir()

    calendar = clean_calendar(synthetic_calendar())
    sales_days = calendar.iloc[:N_DAYS].reset_index(drop=True)
    panel = synthetic_panel(sales_days)

    calendar.to_parquet(processed / "calendar.parquet", index=False)
    panel.to_parquet(processed / "sample_panel.parquet", index=False)

    tail = panel[panel["day_idx"] > N_DAYS - N_TRAIN_HEAD]
    wide = tail.pivot(
        index=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        columns="d",
        values="units",
    ).reset_index()
    wide.columns.name = None
    wide.to_parquet(processed / "series_history_tail.parquet", index=False)

    return processed


def test_chronological_split_positions():
    series = pd.Series(range(1, 71))
    origin, first_validation = chronological_split(series)

    assert origin == 42
    assert first_validation == 43


def test_run_training_produces_artifacts(synthetic_processed: Path, tmp_path: Path):
    models_dir = tmp_path / "models"

    result = run_training(
        max_iter=8, processed_dir=synthetic_processed, models_dir=models_dir
    )

    version_dir = models_dir / result["version"]
    assert (version_dir / "bundle.joblib").is_file()
    assert (version_dir / "metadata.json").is_file()
    assert (version_dir / "evaluation.json").is_file()
    assert (version_dir / "validation_predictions.parquet").is_file()

    evaluation = result["evaluation"]
    assert set(evaluation["model"]["by_horizon"]) == {"7", "14", "28"}
    assert "mean_baseline" in evaluation["baselines"]

    metadata = result["metadata"]
    assert metadata["training"]["train_days"].endswith("d_42")
    assert metadata["features"]["categorical"] == list(CATEGORICAL_FEATURES)


def test_bundle_roundtrip_and_latest_version(synthetic_processed: Path, tmp_path: Path):
    models_dir = tmp_path / "models"
    run_training(max_iter=5, processed_dir=synthetic_processed, models_dir=models_dir)

    version = latest_version(models_dir)
    bundle = load_bundle(version=version, models_dir=models_dir)

    assert bundle["version"] == version
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    assert "model" in bundle


def test_load_bundle_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Models directory not found"):
        load_bundle(models_dir=tmp_path / "nope")


class LagStubModel:
    """Predicts exactly the previous day's value (tests recursion wiring)."""

    def predict(self, X):
        return X["lag_1"].to_numpy(dtype=float)


def test_recursive_forecast_feeds_predictions_back():
    calendar = clean_calendar(synthetic_calendar())
    sales_days = calendar.iloc[:N_DAYS].reset_index(drop=True)
    panel = synthetic_panel(sales_days)

    tail = panel[panel["day_idx"] >= N_DAYS - 40]
    features = build_features(tail)
    bundle = {
        "model": LagStubModel(),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_categories": {
            col: list(features[col].cat.categories) for col in CATEGORICAL_FEATURES
        },
    }

    horizon = 7
    forecast = recursive_forecast(bundle, tail, calendar, horizon)

    assert len(forecast) == 3 * horizon
    assert sorted(forecast["day_idx"].unique().tolist()) == list(
        range(N_DAYS + 1, N_DAYS + 8)
    )

    # Each series has a constant-ish weekly pattern; with the lag-stub the
    # forecast must equal the last observed value on EVERY horizon day —
    # this only holds if predictions are fed back into the lag features.
    for series_id, group in forecast.groupby("id"):
        history_last = tail[tail["id"] == series_id]["units"].iloc[-1]
        assert np.allclose(group["predicted_units"], history_last)


def test_forecast_from_history_uses_processed_tail(
    synthetic_processed: Path, tmp_path: Path
):
    models_dir = tmp_path / "models"
    run_training(max_iter=5, processed_dir=synthetic_processed, models_dir=models_dir)

    bundle = load_bundle(models_dir=models_dir)
    calendar = clean_calendar(synthetic_calendar())

    series_id = "ITEM_A_STORE_1"
    forecast = forecast_from_history(
        bundle, series_id, calendar, horizon=7, processed_dir=synthetic_processed
    )

    assert set(forecast["id"]) == {series_id}
    assert len(forecast) == 7
    assert forecast["predicted_units"].notna().all()
    assert (forecast["predicted_units"] >= 0).all()


def test_forecast_from_history_unknown_series(
    synthetic_processed: Path, tmp_path: Path
):
    models_dir = tmp_path / "models"
    run_training(max_iter=5, processed_dir=synthetic_processed, models_dir=models_dir)
    bundle = load_bundle(models_dir=models_dir)
    calendar = clean_calendar(synthetic_calendar())

    with pytest.raises(KeyError, match="Unknown series_id"):
        forecast_from_history(
            bundle, "MISSING", calendar, horizon=7, processed_dir=synthetic_processed
        )
