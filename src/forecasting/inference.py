"""Recursive multi-horizon inference with a trained model bundle.

The recursive scheme feeds each prediction back into the feature history
instead of using actual future sales, so evaluation matches what is
available at deployment time. Future sell prices are assumed equal to the
last observed price unless the caller supplies explicit future prices.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.pipeline import load_processed, state_snap
from src.features.engineering import (
    FEATURE_COLUMNS,
    REQUIRED_HISTORY_DAYS,  # single source of truth with training features
    build_features,
)

logger = logging.getLogger(__name__)

MODEL_COLUMNS = [
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


def latest_version(models_dir: Path | None = None) -> str:
    """Return the newest model version directory name."""

    models_dir = Path(models_dir) if models_dir else get_settings().models_dir
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"Models directory not found: {models_dir}. "
            "Run `python -m src.forecasting.train` first."
        )

    versions = sorted(
        (path for path in models_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not versions:
        raise FileNotFoundError(
            f"No model versions found in {models_dir}. "
            "Run `python -m src.forecasting.train` first."
        )
    return versions[-1].name


def load_bundle(version: str | None = None, models_dir: Path | None = None) -> dict:
    """Load a trained model bundle (model + feature metadata)."""

    models_dir = Path(models_dir) if models_dir else get_settings().models_dir
    version = version or latest_version(models_dir)
    path = models_dir / version / "bundle.joblib"

    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    bundle = joblib.load(path)
    bundle["version"] = version
    return bundle


def _tail_per_series(panel: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return the last ``days`` rows per series from a long panel."""

    max_day = panel["day_idx"].max()
    cutoff = max_day - days + 1
    return panel[panel["day_idx"] >= cutoff].copy()


def recursive_forecast(
    bundle: dict,
    panel_tail: pd.DataFrame,
    calendar: pd.DataFrame,
    horizon: int,
    future_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast ``horizon`` days for every series in ``panel_tail``.

    Parameters
    ----------
    bundle:
        Loaded model bundle with ``model`` and ``feature_columns``.
    panel_tail:
        Long panel covering at least ``REQUIRED_HISTORY_DAYS`` days per
        series, ending at the forecast origin.
    calendar:
        Full cleaned calendar (must cover the forecast days).
    horizon:
        Number of days to forecast (7, 14, or 28 in the product).
    future_prices:
        Optional DataFrame with columns id, day_idx, sell_price supplying
        planned prices; defaults to the last observed price per series.

    Returns
    -------
    DataFrame with columns id, day_idx, d, date, predicted_units.
    """

    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)

    work = panel_tail[MODEL_COLUMNS].copy()
    origin_day = int(work["day_idx"].max())

    last_price = work.groupby("id")["sell_price"].last()
    last_meta = work.groupby("id", sort=False).first()[
        ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ]

    results: list[pd.DataFrame] = []

    for step in range(1, horizon + 1):
        day_idx = origin_day + step
        day_row = calendar[calendar["day_idx"] == day_idx]
        if len(day_row) != 1:
            raise ValueError(
                f"Calendar has no unique row for day_idx={day_idx}; "
                "cannot extend the forecast."
            )

        appended = pd.DataFrame(
            {
                "id": last_meta.index,
                "item_id": last_meta["item_id"].to_numpy(),
                "dept_id": last_meta["dept_id"].to_numpy(),
                "cat_id": last_meta["cat_id"].to_numpy(),
                "store_id": last_meta["store_id"].to_numpy(),
                "state_id": last_meta["state_id"].to_numpy(),
                "d": day_row["d"].iloc[0],
                "day_idx": day_idx,
                "units": np.nan,
                "sell_price": last_price.to_numpy(),
                "date": day_row["date"].iloc[0],
                "wm_yr_wk": int(day_row["wm_yr_wk"].iloc[0]),
                "wday": int(day_row["wday"].iloc[0]),
                "month": int(day_row["month"].iloc[0]),
                "is_event": int(day_row["is_event"].iloc[0]),
            }
        )

        # SNAP windows are known in advance: select the state-specific flag.
        day_source = pd.DataFrame([day_row.iloc[0]] * len(last_meta)).reset_index(
            drop=True
        )
        appended["snap"] = state_snap(
            day_source, pd.Series(last_meta["state_id"].to_numpy())
        ).to_numpy()

        if future_prices is not None:
            supplied = future_prices[future_prices["day_idx"] == day_idx]
            if len(supplied):
                prices = appended.merge(
                    supplied[["id", "sell_price"]],
                    on="id",
                    how="left",
                    suffixes=("", "_new"),
                )
                appended["sell_price"] = prices["sell_price_new"].fillna(
                    prices["sell_price"]
                )

        work = pd.concat([work, appended], ignore_index=True)
        work = _tail_per_series(work, REQUIRED_HISTORY_DAYS)

        features = build_features(work)

        # Cast categoricals to the exact training categories so unseen
        # levels map safely instead of crashing the encoder.
        for column, categories in bundle.get("categorical_categories", {}).items():
            features[column] = pd.Categorical(features[column], categories=categories)

        current = features[features["day_idx"] == day_idx]
        current = current.set_index("id").reindex(last_meta.index)

        predicted = np.clip(model.predict(current[feature_columns]), 0.0, None)

        # Feed the prediction back into the history (recursive scheme).
        mask = work["day_idx"] == day_idx
        work.loc[mask, "units"] = predicted

        results.append(
            pd.DataFrame(
                {
                    "id": last_meta.index,
                    "day_idx": day_idx,
                    "d": appended["d"].to_numpy(),
                    "date": appended["date"].to_numpy(),
                    "predicted_units": predicted,
                }
            )
        )

    return pd.concat(results, ignore_index=True)


def load_calendar(processed_dir: Path | None = None) -> pd.DataFrame:
    """Load the processed cleaned calendar."""

    processed_dir = (
        Path(processed_dir) if processed_dir else get_settings().processed_data_dir
    )
    return load_processed("calendar", processed_dir)


def forecast_from_history(
    bundle: dict,
    series_id: str,
    calendar: pd.DataFrame,
    horizon: int,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Forecast one series using its stored history tail.

    Loads the recent history of ``series_id`` from the processed
    ``series_history_tail`` dataset (covers all 30,490 series).
    """

    processed_dir = (
        Path(processed_dir) if processed_dir else get_settings().processed_data_dir
    )
    tail = load_processed("series_history_tail", processed_dir)

    series = tail[tail["id"] == series_id]
    if series.empty:
        raise KeyError(f"Unknown series_id: {series_id}")

    day_columns = sorted(
        (col for col in series.columns if col.startswith("d_")),
        key=lambda col: int(col[2:]),
    )

    long = series.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=day_columns,
        var_name="d",
        value_name="units",
    )
    long["day_idx"] = long["d"].str[2:].astype(int)

    # Reattach calendar fields (future price = last observed at inference).
    calendar_fields = calendar[
        [
            "d",
            "date",
            "wm_yr_wk",
            "wday",
            "month",
            "is_event",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
    ]
    long = long.merge(calendar_fields, on="d", how="left", validate="many_to_one")

    long["snap"] = state_snap(long, long["state_id"]).to_numpy()
    long["sell_price"] = np.nan  # unknown per-day price; inference back-fills
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    return recursive_forecast(bundle, long, calendar, horizon)
