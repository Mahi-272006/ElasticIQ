"""Shared pytest fixtures.

Every fixture builds small, realistic M5-shaped data so that unit tests never
need the full 240 MB dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

N_DAYS = 35

METADATA_ROWS = [
    # id, item_id, dept_id, cat_id, store_id, state_id
    ("FOODS_3_090_CA_1", "FOODS_3_090", "FOODS_3", "FOODS", "CA_1", "CA"),
    ("FOODS_3_090_TX_1", "FOODS_3_090", "FOODS_3", "FOODS", "TX_1", "TX"),
    ("HOBBIES_1_001_CA_1", "HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA"),
]

# Deterministic non-negative sales values including some zero days.
SALES_MATRIX = [
    [
        0,
        1,
        2,
        0,
        3,
        1,
        0,
        2,
        4,
        1,
        0,
        0,
        2,
        3,
        1,
        0,
        2,
        1,
        0,
        3,
        2,
        1,
        0,
        2,
        1,
        0,
        3,
        1,
        2,
        0,
        1,
        2,
        0,
        1,
        3,
    ],
    [
        2,
        3,
        1,
        2,
        0,
        1,
        2,
        3,
        1,
        0,
        2,
        1,
        3,
        0,
        2,
        1,
        1,
        0,
        2,
        3,
        1,
        2,
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        1,
        0,
        2,
        3,
        1,
        0,
    ],
    [
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
    ],
]


@pytest.fixture
def tiny_calendar() -> pd.DataFrame:
    """A calendar covering N_DAYS starting at the real M5 origin date."""

    dates = pd.date_range("2011-01-29", periods=N_DAYS, freq="D")
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
            "wm_yr_wk": [11101 + index // 7 for index in range(N_DAYS)],
            "weekday": [weekday_names[index % 7] for index in range(N_DAYS)],
            "wday": [index % 7 + 1 for index in range(N_DAYS)],
            "month": dates.month,
            "year": dates.year,
            "snap_CA": [1 if index % 3 == 0 else 0 for index in range(N_DAYS)],
            "snap_TX": [1 if index % 4 == 0 else 0 for index in range(N_DAYS)],
            "snap_WI": [1 if index % 5 == 0 else 0 for index in range(N_DAYS)],
            "d": [f"d_{index + 1}" for index in range(N_DAYS)],
        }
    )

    # Blank events are legitimate: only a few days carry an event.
    frame["event_name_1"] = pd.NA
    frame["event_type_1"] = pd.NA
    frame["event_name_2"] = pd.NA
    frame["event_type_2"] = pd.NA
    frame.loc[10, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]
    frame.loc[20, ["event_name_1", "event_type_1"]] = ["Easter", "Religious"]

    return frame


@pytest.fixture
def tiny_sales(tiny_calendar: pd.DataFrame) -> pd.DataFrame:
    """A tiny wide-format sales frame with deterministic values."""

    frame = pd.DataFrame(
        METADATA_ROWS,
        columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )

    for index in range(N_DAYS):
        frame[f"d_{index + 1}"] = [row[index] for row in SALES_MATRIX]

    assert list(frame.columns[:6]) == [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]
    assert tiny_calendar["d"].iloc[-1] == f"d_{N_DAYS}"
    return frame


@pytest.fixture
def tiny_prices(tiny_calendar: pd.DataFrame) -> pd.DataFrame:
    """Weekly prices for the fixture items, matching calendar weeks."""

    weeks = sorted(tiny_calendar["wm_yr_wk"].unique())
    rows = []

    for store_id, item_id in [
        ("CA_1", "FOODS_3_090"),
        ("TX_1", "FOODS_3_090"),
        ("CA_1", "HOBBIES_1_001"),
    ]:
        base = 2.5 if item_id.startswith("FOODS") else 5.0
        for offset, week in enumerate(weeks):
            rows.append(
                {
                    "store_id": store_id,
                    "item_id": item_id,
                    "wm_yr_wk": week,
                    "sell_price": round(base + 0.1 * offset, 2),
                }
            )

    return pd.DataFrame(rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])


@pytest.fixture
def tiny_submission(tiny_sales: pd.DataFrame) -> pd.DataFrame:
    """A submission frame with both validation and evaluation id suffixes."""

    ids = tiny_sales["id"].tolist()
    return pd.DataFrame(
        {"id": ids + [f"{identifier}_evaluation" for identifier in ids]}
    )


@pytest.fixture
def raw_dir(
    tmp_path: Path, tiny_calendar, tiny_sales, tiny_prices, tiny_submission
) -> Path:
    """Write all fixture frames as M5-shaped CSVs in a temp directory."""

    raw = tmp_path / "raw"
    raw.mkdir()

    tiny_calendar.to_csv(raw / "calendar.csv", index=False)
    tiny_sales.to_csv(raw / "sales_train_validation.csv", index=False)
    tiny_sales.to_csv(raw / "sales_train_evaluation.csv", index=False)
    tiny_prices.to_csv(raw / "sell_prices.csv", index=False)
    tiny_submission.to_csv(raw / "sample_submission.csv", index=False)

    return raw


# ---------------------------------------------------------------------------
# API test fixtures: processed Parquet frames + stub model + isolated SQLite
# ---------------------------------------------------------------------------

N_API_HISTORY = 60  # >= features REQUIRED_HISTORY_DAYS
N_API_CALENDAR = 75  # history + forecast room
N_GENERATED_SERIES = 117  # + 3 named rows -> 120 series (drift needs >= 100)


def _api_calendar() -> pd.DataFrame:
    """Cleaned-calendar shaped frame covering the API fixture window."""

    dates = pd.date_range("2011-01-29", periods=N_API_CALENDAR, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": [11101 + index // 7 for index in range(N_API_CALENDAR)],
            "wday": [index % 7 + 1 for index in range(N_API_CALENDAR)],
            "month": dates.month,
            "d": [f"d_{index + 1}" for index in range(N_API_CALENDAR)],
            "day_idx": list(range(1, N_API_CALENDAR + 1)),
            "is_event": [
                1 if index % 15 == 0 else 0 for index in range(N_API_CALENDAR)
            ],
            "snap_CA": [1 if index % 3 == 0 else 0 for index in range(N_API_CALENDAR)],
            "snap_TX": [1 if index % 4 == 0 else 0 for index in range(N_API_CALENDAR)],
            "snap_WI": [1 if index % 5 == 0 else 0 for index in range(N_API_CALENDAR)],
        }
    )
    return frame


def _api_products_and_tail() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Catalog frame plus wide history tail (120 series, deterministic)."""

    products = pd.DataFrame(
        METADATA_ROWS,
        columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )

    for index in range(N_GENERATED_SERIES):
        products.loc[len(products)] = [
            f"GEN_{index:04d}_CA_2",
            f"GEN_{index:04d}",
            "GEN_1",
            "FOODS",
            "CA_2",
            "CA",
        ]

    day_cols = [f"d_{day}" for day in range(1, N_API_HISTORY + 1)]
    tail = products.copy()
    for day in range(N_API_HISTORY):
        tail[f"d_{day + 1}"] = [
            (series_index + day) % 5 for series_index in range(len(products))
        ]

    assert list(tail.columns[6:]) == day_cols
    return products, tail


def _api_price_stats(products: pd.DataFrame) -> pd.DataFrame:
    """Per-series price statistics (first > last: a real downward drift)."""

    rows = []
    for index, row in products.iterrows():
        first = 2.0 + (index % 7) * 0.1
        last = first * 0.95
        rows.append(
            {
                "store_id": row["store_id"],
                "item_id": row["item_id"],
                "weeks_priced": 10,
                "first_wm_yr_wk": 11101,
                "last_wm_yr_wk": 11110,
                "first_price": round(first, 2),
                "last_price": round(last, 2),
                "min_price": round(last - 0.1, 2),
                "max_price": round(first + 0.1, 2),
                "mean_price": round((first + last) / 2, 2),
                "std_price": 0.05,
                "price_changes": 2,
            }
        )
    return pd.DataFrame(rows)


def _api_aggregates(tail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily total/category/store aggregates derived from the fixture tail."""

    calendar = _api_calendar().set_index("d")["date"]
    day_cols = [col for col in tail.columns if col.startswith("d_")]

    total = tail[day_cols].sum(axis=0).rename("units").reset_index()
    total.columns = ["d", "units"]
    total["date"] = total["d"].map(calendar)

    def _grouped(group_col: str, key: str) -> pd.DataFrame:
        stacked = tail.groupby(group_col, sort=True)[day_cols].sum().stack()
        frame = stacked.rename("units").reset_index()
        frame.columns = [key, "d", "units"]
        frame["date"] = frame["d"].map(calendar)
        return frame

    return {
        "aggregates/daily_total_units": total,
        "aggregates/daily_category_units": _grouped("cat_id", "cat_id"),
        "aggregates/daily_store_units": _grouped("store_id", "store_id"),
    }


class StubModel:
    """Deterministic stand-in for the trained HGB regressor."""

    def predict(self, features):
        import numpy as np

        return np.full(len(features), 3.0)


@pytest.fixture
def api_processed_dir(tmp_path: Path) -> Path:
    """Processed-data directory with minimal M5-shaped Parquet frames."""

    processed = tmp_path / "processed"
    processed.mkdir()

    products, tail = _api_products_and_tail()

    _api_calendar().to_parquet(processed / "calendar.parquet", index=False)
    products.to_parquet(processed / "products.parquet", index=False)
    tail.to_parquet(processed / "series_history_tail.parquet", index=False)
    _api_price_stats(products).to_parquet(
        processed / "price_stats.parquet", index=False
    )

    aggregates = _api_aggregates(tail)
    (processed / "aggregates").mkdir()
    for name, frame in aggregates.items():
        frame.to_parquet(processed / f"{name}.parquet", index=False)

    return processed


@pytest.fixture
def api_models_dir(tmp_path: Path) -> Path:
    """Models directory with one stub version (metadata + evaluation)."""

    import json

    version_dir = tmp_path / "models" / "stub_v1"
    version_dir.mkdir(parents=True)

    metadata = {
        "version": "stub_v1",
        "algorithm": "stub",
        "trained_at_utc": "2026-01-01T00:00:00+00:00",
    }
    evaluation = {
        "model": {
            "overall": {"wape_percent": 50.0, "mae_units": 1.0, "rows": 100},
            "by_horizon": {"7": {"wape_percent": 45.0, "mae_units": 0.9, "rows": 20}},
        },
        "baselines": {
            "mean_baseline": {"overall": {"wape_percent": 55.0}, "by_horizon": {}}
        },
        "comparison_note": "stub evaluation for API tests",
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (version_dir / "evaluation.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    (version_dir / "bundle.joblib").write_bytes(b"stub")  # never loaded in tests

    return tmp_path / "models"


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch, api_processed_dir, api_models_dir):
    """TestClient against a fully isolated app (tmp data + tmp SQLite)."""

    from fastapi.testclient import TestClient

    from src.api.app import create_app
    from src.api.services import AppState
    from src.db.session import get_settings, reset_engine
    from src.features.engineering import FEATURE_COLUMNS

    monkeypatch.setenv(
        "ELASTICIQ_DATABASE_URL",
        f"sqlite:///{(tmp_path / 'api_test.db').as_posix()}",
    )
    get_settings.cache_clear()
    reset_engine()

    state = AppState(processed_dir=api_processed_dir, models_dir=api_models_dir)
    state._bundle = {
        "model": StubModel(),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_categories": {},
        "version": "stub_v1",
    }

    app = create_app(state)
    with TestClient(app) as client:
        yield client

    reset_engine()
    get_settings.cache_clear()
