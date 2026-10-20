"""Tests for the preprocessing pipeline (fixture-based; no M5 data needed)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pipeline import (
    attach_calendar_fields,
    attach_prices,
    check_calendar_sales_mapping,
    clean_calendar,
    compute_aggregates,
    load_processed,
    melt_sales_long,
    run_pipeline,
    summarize_prices_chunked,
)
from tests.conftest import N_DAYS, SALES_MATRIX


def test_clean_calendar_adds_types_and_event_flags(tiny_calendar: pd.DataFrame):
    cleaned = clean_calendar(tiny_calendar)

    assert cleaned["date"].dtype.kind == "M"
    assert cleaned["is_event"].sum() == 2  # SuperBowl + Easter fixture days
    assert cleaned["day_idx"].tolist() == list(range(1, N_DAYS + 1))


def test_mapping_check_passes(tiny_calendar: pd.DataFrame, tiny_sales: pd.DataFrame):
    day_cols = [col for col in tiny_sales.columns if col.startswith("d_")]
    result = check_calendar_sales_mapping(tiny_calendar, day_cols)

    assert result["passed"] is True
    assert result["first_date"] == "2011-01-29"
    assert result["sales_day_columns"] == N_DAYS


def test_mapping_check_fails_on_missing_day(
    tiny_calendar: pd.DataFrame, tiny_sales: pd.DataFrame
):
    day_cols = [col for col in tiny_sales.columns if col.startswith("d_")]
    broken = tiny_calendar.iloc[:-1]

    result = check_calendar_sales_mapping(broken, day_cols)
    assert result["passed"] is False
    assert result["sales_days_missing_from_calendar"] == [f"d_{N_DAYS}"]


def test_melt_produces_sorted_long_frame(tiny_sales: pd.DataFrame):
    day_cols = [f"d_{index}" for index in range(1, N_DAYS + 1)]
    long = melt_sales_long(tiny_sales, day_cols)

    assert len(long) == 3 * N_DAYS
    # Sorted by (id, day_idx): first series first, days in order.
    first_id = long["id"].iloc[0]
    assert long["id"].iloc[:N_DAYS].eq(first_id).all()
    assert long["day_idx"].iloc[:N_DAYS].tolist() == list(range(1, N_DAYS + 1))


def test_melt_preserves_values(tiny_sales: pd.DataFrame):
    day_cols = [f"d_{index}" for index in range(1, N_DAYS + 1)]
    long = melt_sales_long(tiny_sales, day_cols)

    first_id = tiny_sales["id"].iloc[0]
    values = long.loc[long["id"] == first_id, "units"].to_numpy()
    expected = tiny_sales.loc[tiny_sales["id"] == first_id, day_cols].to_numpy().ravel()

    assert np.array_equal(values, expected)


def test_attach_prices_joins_on_week(tiny_sales, tiny_calendar, tiny_prices):
    day_cols = [f"d_{index}" for index in range(1, N_DAYS + 1)]
    cleaned = clean_calendar(tiny_calendar)
    long = attach_calendar_fields(melt_sales_long(tiny_sales, day_cols), cleaned)
    long = attach_prices(long, tiny_prices)

    # All fixture store/item pairs are priced for every week, so the join
    # must fill every row (a left-join miss would show NaN here).
    assert long["sell_price"].notna().all()


def test_aggregates_match_series_totals(
    tiny_sales: pd.DataFrame, tiny_calendar: pd.DataFrame
):
    day_cols = [f"d_{index}" for index in range(1, N_DAYS + 1)]
    aggregates = compute_aggregates(tiny_sales, clean_calendar(tiny_calendar), day_cols)

    total = aggregates["aggregates/daily_total_units"]
    assert int(total["units"].iloc[0]) == int(tiny_sales["d_1"].sum())

    # Category breakdown must sum back to the total on every date.
    by_category = aggregates["aggregates/daily_category_units"]
    combined = by_category.groupby("d")["units"].sum()
    expected = total.set_index("d")["units"]
    assert (
        combined.reindex(expected.index).to_numpy().tolist()
        == expected.to_numpy().tolist()
    )


def test_price_summary_counts_cross_boundary_changes(
    tmp_path: Path, tiny_prices: pd.DataFrame
):
    path = tmp_path / "sell_prices.csv"
    tiny_prices.to_csv(path, index=False)

    # Chunk size 2 forces the same key to span multiple chunks; the
    # change counter must still count every consecutive change.
    stats, report = summarize_prices_chunked(path, chunk_size=2)

    foods = stats[
        (stats["item_id"] == "FOODS_3_090") & (stats["store_id"] == "CA_1")
    ].iloc[0]

    weeks = N_DAYS // 7
    assert foods["weeks_priced"] == weeks
    assert foods["price_changes"] == weeks - 1  # rising price every week
    assert foods["max_price"] > foods["min_price"]
    assert report["price_rows"] == len(tiny_prices)


def test_run_pipeline_end_to_end(raw_dir: Path, tmp_path: Path):
    out_dir = tmp_path / "processed"
    report = run_pipeline(sample_series=2, raw_dir=raw_dir, out_dir=out_dir)

    assert report["calendar_mapping"]["passed"] is True
    assert report["sample_series"] == 2
    assert report["panel_rows"] == 2 * N_DAYS
    assert report["total_series"] == 3
    assert report["price_report"]["price_weeks_missing_from_calendar"] == []
    assert report["price_report"]["price_keys_without_product"] == 0

    panel = load_processed("sample_panel", out_dir)
    assert len(panel) == 2 * N_DAYS
    assert {"units", "sell_price", "date", "cat_id"} <= set(panel.columns)

    tail = load_processed("series_history_tail", out_dir)
    assert len(tail) == 3  # ALL series, not just the sample

    total = load_processed("aggregates/daily_total_units", out_dir)
    expected_units = sum(sum(row) for row in SALES_MATRIX)
    assert int(total["units"].sum()) == expected_units

    saved = json.loads((out_dir / "pipeline_report.json").read_text(encoding="utf-8"))
    assert saved["sample_series"] == 2
