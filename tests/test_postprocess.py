"""Tests for the final-forecast post-processing stage.

The shipped system is ``scale[id] x level[id] x shape[id, day]``:

- ``level``   = mean of the last ``level_window`` observed days of history,
- ``shape``   = the raw model output normalised to mean 1 within a series,
- ``scale``   = per-series factor fitted leak-free on a held-out window.

These tests pin the arithmetic, the degenerate cases (all-zero raw output,
zero recent level, series absent from history), the fitting contract, and
the wiring through ``recursive_forecast`` and ``run_training``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import clean_calendar, state_snap
from src.features.engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    build_features,
)
from src.forecasting.inference import load_bundle, recursive_forecast
from src.forecasting.postprocess import (
    DEFAULT_LEVEL_WINDOW,
    DEFAULT_SHRINK,
    apply_postprocess,
    fit_postprocess,
    levelled_forecast,
)
from src.forecasting.train import run_training

# ---------------------------------------------------------------------------
# Hand-built frames: two series, 20 history days, 3 forecast days
# ---------------------------------------------------------------------------


def history_frame(level_a: float = 4.0, level_b: float = 2.0) -> pd.DataFrame:
    rows = [
        {"id": series_id, "day_idx": day, "units": level}
        for day in range(1, 21)
        for series_id, level in (("A", level_a), ("B", level_b))
    ]
    return pd.DataFrame(rows)


def raw_frame(rows: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": series_id,
                "day_idx": day,
                "d": f"d_{day}",
                "date": "2011-02-18",
                "predicted_units": value,
            }
            for series_id, values in rows.items()
            for day, value in zip(
                range(21, 21 + len(values)), values, strict=True
            )
        ]
    )


def test_levelled_forecast_matches_hand_computation():
    raw = raw_frame({"A": [1.0, 3.0, 2.0], "B": [1.0, 2.0, 3.0]})

    out = levelled_forecast(raw, history_frame(), level_window=7)

    # A: level 4 x shape [0.5, 1.5, 1.0]; B: level 2 x shape [0.5, 1, 1.5].
    a = out[out["id"] == "A"]["predicted_units"].to_numpy()
    b = out[out["id"] == "B"]["predicted_units"].to_numpy()
    assert np.allclose(a, [2.0, 6.0, 4.0])
    assert np.allclose(b, [1.0, 2.0, 3.0])


def test_level_only_uses_last_level_window_days():
    history = history_frame()
    # Fresh week is hot for A (days 14..20), the rest of history is cold.
    history.loc[
        (history["id"] == "A") & (history["day_idx"] > 13), "units"
    ] = 9.0
    raw = raw_frame({"A": [2.0, 2.0, 2.0], "B": [1.0, 1.0, 1.0]})

    fresh = levelled_forecast(raw, history, level_window=7)
    stale = levelled_forecast(raw, history, level_window=20)

    assert np.allclose(fresh[fresh["id"] == "A"]["predicted_units"], 9.0)
    # 13 cold days at 4.0 plus 7 hot days at 9.0 -> mean 5.75.
    assert np.allclose(stale[stale["id"] == "A"]["predicted_units"], 5.75)


def test_all_zero_raw_predictions_stay_zero():
    raw = raw_frame({"A": [0.0, 0.0, 0.0], "B": [1.0, 1.0, 1.0]})

    out = levelled_forecast(raw, history_frame(), level_window=7)

    a = out[out["id"] == "A"]["predicted_units"].to_numpy()
    assert np.all(a == 0.0)  # a flat zero forecast must not become a level


def test_zero_recent_level_forecasts_zero():
    raw = raw_frame({"A": [1.0, 2.0, 3.0], "B": [1.0, 1.0, 1.0]})

    out = levelled_forecast(raw, history_frame(level_b=0.0), level_window=7)

    assert np.all(out[out["id"] == "B"]["predicted_units"] == 0.0)
    # A is unaffected: level 4 x shape [0.5, 1, 1.5].
    assert np.allclose(out[out["id"] == "A"]["predicted_units"], [2.0, 4.0, 6.0])


def test_series_missing_from_history_keeps_model_level():
    raw = raw_frame({"A": [1.0, 3.0, 2.0], "Z": [2.0, 4.0, 6.0]})

    out = levelled_forecast(raw, history_frame(), level_window=7)

    z = out[out["id"] == "Z"]["predicted_units"].to_numpy()
    assert np.allclose(z, [2.0, 4.0, 6.0])  # falls back to the model's own level


def test_invalid_level_window_raises():
    raw = raw_frame({"A": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="level_window"):
        levelled_forecast(raw, history_frame(), level_window=0)


def test_apply_postprocess_uses_series_and_global_scale():
    raw = raw_frame({"A": [1.0, 3.0, 2.0], "B": [1.0, 2.0, 3.0]})
    postprocess = {
        "level_window": 7,
        "per_series": {"A": 2.0},
        "global_scale": 0.5,
    }

    out = apply_postprocess(raw, history_frame(), postprocess)

    a = out[out["id"] == "A"]["predicted_units"].to_numpy()
    b = out[out["id"] == "B"]["predicted_units"].to_numpy()
    assert np.allclose(a, [4.0, 12.0, 8.0])  # 2.0 x levelled A
    assert np.allclose(b, [0.5, 1.0, 1.5])  # 0.5 x levelled B


def test_apply_postprocess_preserves_schema():
    raw = raw_frame({"A": [1.0, 3.0, 2.0], "B": [1.0, 2.0, 3.0]})
    postprocess = {"level_window": 7, "per_series": {"A": 2.0}, "global_scale": 1.0}

    out = apply_postprocess(raw, history_frame(), postprocess)

    assert list(out.columns) == list(raw.columns)
    assert len(out) == len(raw)
    assert (out["predicted_units"] >= 0).all()


def constant_frame(
    values: dict[str, float], start: int, n: int, column: str
) -> pd.DataFrame:
    """``n`` identical rows per series starting at day ``start``."""

    keys = ["id", "day_idx", "d", "date", column]
    frame = pd.DataFrame(
        [
            {
                "id": series_id,
                "day_idx": day,
                "d": f"d_{day}",
                "date": "2011-02-18",
                column: value,
            }
            for series_id, value in values.items()
            for day in range(start, start + n)
        ]
    )
    assert list(frame.columns) == keys
    return frame


def test_fit_postprocess_recovers_known_scale():
    history = history_frame(4.0, 2.0)  # ends at day 20 (pre-window history)
    # 7 rows per series so the minimum-row guard still fits per-series scales.
    fit_raw = constant_frame({"A": 4.0, "B": 2.0}, start=21, n=7,
                             column="predicted_units")
    actuals = constant_frame({"A": 8.0, "B": 1.0}, start=21, n=7,
                             column="actual")

    postprocess = fit_postprocess(
        fit_raw,
        history,
        actuals,
        level_window=7,
        shrink=0.0,
        fitted_on={"scale_fit_days": "d_21..d_27"},
    )

    # Constant raw == its own mean, so levelled prior == level: A is exactly
    # 2x under-predicted, B exactly 2x over-predicted.
    assert postprocess["per_series"]["A"] == pytest.approx(2.0)
    assert postprocess["per_series"]["B"] == pytest.approx(0.5)
    assert postprocess["fitted_series"] == 2
    assert postprocess["fitted_on"]["scale_fit_days"] == "d_21..d_27"
    assert postprocess["level_window"] == 7

    eval_raw = constant_frame({"A": 4.0, "B": 2.0}, start=28, n=3,
                              column="predicted_units")
    applied = apply_postprocess(eval_raw, history, postprocess)
    assert np.allclose(applied[applied["id"] == "A"]["predicted_units"], 8.0)
    assert np.allclose(applied[applied["id"] == "B"]["predicted_units"], 1.0)


def test_fit_postprocess_rejects_disjoint_windows():
    history = history_frame()
    fit_raw = raw_frame({"A": [4.0] * 3, "B": [2.0] * 3})
    actuals = pd.DataFrame(
        [{"id": "A", "day_idx": 999, "actual": 1.0}]
    )

    with pytest.raises(ValueError, match="overlapping"):
        fit_postprocess(fit_raw, history, actuals)


def test_fit_postprocess_rejects_empty_predictions():
    history = history_frame()
    empty = raw_frame({"A": [1.0]}).iloc[0:0]
    actuals = pd.DataFrame([{"id": "A", "day_idx": 21, "actual": 1.0}])

    with pytest.raises(ValueError, match="overlapping"):
        fit_postprocess(empty, history, actuals)


# ---------------------------------------------------------------------------
# Wiring: recursive_forecast applies the block, training ships it
# ---------------------------------------------------------------------------


class LagStubModel:
    """Predicts exactly the previous day's value (recursion wiring probe)."""

    def predict(self, X):
        return X["lag_1"].to_numpy(dtype=float)


def test_recursive_forecast_applies_postprocess_identically_to_helper():
    calendar = clean_calendar(build_calendar(CALENDAR_DAYS))
    sales_days = calendar.iloc[:N_SALES_DAYS].reset_index(drop=True)
    panel = build_panel(sales_days)
    tail = panel[panel["day_idx"] >= N_SALES_DAYS - 40]

    features = build_features(tail)
    bundle = {
        "model": LagStubModel(),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_categories": {
            col: list(features[col].cat.categories) for col in CATEGORICAL_FEATURES
        },
    }
    postprocess = {
        "level_window": DEFAULT_LEVEL_WINDOW,
        "shrink": 0.0,
        "per_series": {series_id: 2.0 for series_id in panel["id"].unique()},
        "global_scale": 0.5,
    }

    raw = recursive_forecast(bundle, tail, calendar, horizon=7)
    expected = apply_postprocess(raw, tail, postprocess)

    served_bundle = {**bundle, "postprocess": postprocess}
    served = recursive_forecast(served_bundle, tail, calendar, horizon=7)

    assert list(served.columns) == list(raw.columns)
    assert (served["id"] == expected["id"]).all()
    assert (served["day_idx"] == expected["day_idx"]).all()
    assert np.allclose(served["predicted_units"], expected["predicted_units"])
    # Sanity: the postprocess really changed the numbers (level x shape != raw).
    assert not np.allclose(served["predicted_units"], raw["predicted_units"])


# ---------------------------------------------------------------------------
# Synthetic processed panel long enough for the inner cross-fit
# ---------------------------------------------------------------------------

CALENDAR_DAYS = 140  # calendar must outlive the sales window (like the M5)
N_SALES_DAYS = 110  # origin d_82, inner cross-fit origin d_54 (> d_28)
N_SHORT_DAYS = 70  # too short to calibrate (inner origin would be d_14)


def build_calendar(n_days: int) -> pd.DataFrame:
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")
    names = [
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
            "weekday": [names[i % 7] for i in range(n_days)],
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


def build_panel(sales_days: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for scale, (item, store, state, cat, dept) in enumerate(
        [
            ("ITEM_A", "STORE_1", "CA", "CAT_A", "DEPT_1"),
            ("ITEM_B", "STORE_1", "CA", "CAT_B", "DEPT_2"),
            ("ITEM_C", "STORE_2", "TX", "CAT_A", "DEPT_1"),
        ],
        start=1,
    ):
        base = 3.0 + 2.0 * scale
        for index, day in enumerate(sales_days["d"]):
            weekly = 1.0 if index % 7 < 2 else 0.0
            units = base + weekly + rng.normal(0, 0.4)
            rows.append(
                {
                    "id": f"{item}_{store}",
                    "item_id": item,
                    "dept_id": dept,
                    "cat_id": cat,
                    "store_id": store,
                    "state_id": state,
                    "d": day,
                    "day_idx": index + 1,
                    "units": max(float(np.round(units)), 0.0),
                    "sell_price": 2.5 + 0.1 * scale,
                    "date": sales_days["date"].iloc[index],
                    "wm_yr_wk": int(sales_days["wm_yr_wk"].iloc[index]),
                    "wday": int(sales_days["wday"].iloc[index]),
                    "month": int(sales_days["month"].iloc[index]),
                    "is_event": 0,
                }
            )

    frame = pd.DataFrame(rows)
    cleaned = clean_calendar(build_calendar(CALENDAR_DAYS)).set_index("d")
    source = cleaned.loc[frame["d"]].reset_index(drop=True)
    frame["snap"] = state_snap(source, frame["state_id"]).to_numpy()
    return frame


def write_processed(path: Path, n_sales_days: int) -> Path:
    processed = path / "processed"
    processed.mkdir(parents=True)

    calendar = clean_calendar(build_calendar(CALENDAR_DAYS))
    sales_days = calendar.iloc[:n_sales_days].reset_index(drop=True)
    panel = build_panel(sales_days)

    calendar.to_parquet(processed / "calendar.parquet", index=False)
    panel.to_parquet(processed / "sample_panel.parquet", index=False)
    return processed


@pytest.fixture
def calibrated_processed(tmp_path: Path) -> Path:
    return write_processed(tmp_path, N_SALES_DAYS)


@pytest.fixture
def short_processed(tmp_path: Path) -> Path:
    return write_processed(tmp_path, N_SHORT_DAYS)


def test_run_training_with_calibrate_ships_postprocess(
    calibrated_processed: Path, tmp_path: Path
):
    models_dir = tmp_path / "models"

    result = run_training(
        max_iter=5,
        processed_dir=calibrated_processed,
        models_dir=models_dir,
        model_version="post_v1",
        calibrate=True,
    )

    assert result["version"] == "post_v1"
    version_dir = models_dir / "post_v1"

    bundle = load_bundle(version="post_v1", models_dir=models_dir)
    postprocess = bundle["postprocess"]
    assert postprocess["level_window"] == DEFAULT_LEVEL_WINDOW
    assert postprocess["shrink"] == DEFAULT_SHRINK
    assert set(postprocess["per_series"]) == {"ITEM_A_STORE_1", "ITEM_B_STORE_1",
                                              "ITEM_C_STORE_2"}
    # Inner cross-fit provenance: origin d_82 -> inner train d_1..d_54,
    # scales fitted on d_55..d_82, validation untouched at d_83..d_110.
    assert postprocess["fitted_on"]["scale_fit_days"] == "d_55..d_82"
    assert postprocess["fitted_on"]["inner_model_train_days"] == "d_1..d_54"

    evaluation = json.loads((version_dir / "evaluation.json").read_text("utf-8"))
    assert set(evaluation["model"]["by_horizon"]) == {"7", "14", "28"}
    assert set(evaluation["raw_model"]["by_horizon"]) == {"7", "14", "28"}
    assert evaluation["raw_model"]["overall"]["rows"] == 3 * 28

    metadata = json.loads((version_dir / "metadata.json").read_text("utf-8"))
    assert metadata["postprocess"]["level_window"] == DEFAULT_LEVEL_WINDOW
    assert metadata["postprocess"]["fitted_rows"] == 3 * 28

    saved = pd.read_parquet(version_dir / "validation_predictions.parquet")
    assert (saved["predicted_units"] >= 0).all()

    # The serving path (bundle with postprocess) must reproduce the saved
    # validation numbers exactly - evaluation and deployment agree.
    panel = pd.read_parquet(calibrated_processed / "sample_panel.parquet")
    calendar = pd.read_parquet(calibrated_processed / "calendar.parquet")
    tail = panel[panel["day_idx"].between(82 - 60 + 1, 82)]
    again = recursive_forecast(bundle, tail, calendar, horizon=28)
    merged = again.merge(
        saved[["id", "day_idx", "predicted_units"]],
        on=["id", "day_idx"],
        suffixes=("_served", "_saved"),
        validate="one_to_one",
    )
    assert np.allclose(
        merged["predicted_units_served"], merged["predicted_units_saved"]
    )


def test_run_training_calibrate_rejects_short_panel(short_processed: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="too short"):
        run_training(
            max_iter=5,
            processed_dir=short_processed,
            models_dir=tmp_path / "models",
            calibrate=True,
        )


def test_run_training_without_calibrate_has_no_postprocess(
    short_processed: Path, tmp_path: Path
):
    models_dir = tmp_path / "models"

    result = run_training(
        max_iter=5, processed_dir=short_processed, models_dir=models_dir
    )

    version_dir = models_dir / result["version"]
    bundle = load_bundle(version=result["version"], models_dir=models_dir)
    assert "postprocess" not in bundle

    evaluation = json.loads((version_dir / "evaluation.json").read_text("utf-8"))
    assert "raw_model" not in evaluation
    assert "model" in evaluation and "baselines" in evaluation

    metadata = json.loads((version_dir / "metadata.json").read_text("utf-8"))
    assert metadata["postprocess"] is None
