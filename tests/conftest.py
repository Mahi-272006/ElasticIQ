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
