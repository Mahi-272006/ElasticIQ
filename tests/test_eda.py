"""Tests for EDA analysis and the report orchestrator (fixture-based)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pipeline import run_pipeline
from src.eda.analysis import (
    compute_eda_summary,
    day_of_week_pattern,
    demand_distribution,
    detect_spikes,
    event_effect,
    temporal_summary,
)
from src.eda.run import run_eda
from tests.conftest import N_DAYS


def _daily_total_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2011-01-29", periods=N_DAYS, freq="D"),
            "units": np.arange(1, N_DAYS + 1) * 100,
        }
    )


def test_temporal_summary_counts_days():
    summary = temporal_summary(_daily_total_frame())

    assert summary["days"] == N_DAYS
    assert summary["start_date"] == "2011-01-29"
    assert summary["total_units"] == int(np.arange(1, N_DAYS + 1).sum() * 100)


def test_day_of_week_covers_all_weekdays(
    tiny_calendar: pd.DataFrame, tiny_sales: pd.DataFrame
):
    daily_total = pd.DataFrame(
        {
            "date": tiny_calendar["date"],
            "units": [100 + index for index in range(N_DAYS)],
        }
    )
    pattern = day_of_week_pattern(daily_total, tiny_calendar)

    assert len(pattern) == 7
    assert {"weekday", "mean", "wday"} <= set(pattern.columns)


def test_detect_spikes_flags_obvious_outlier():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2011-01-29", periods=20, freq="D"),
            "units": [100] * 19 + [10_000],
        }
    )
    result = detect_spikes(frame)

    assert len(result["spike_days"]) == 1
    assert result["spike_days"][0]["units"] == 10_000
    assert result["spike_days"][0]["zscore"] >= 4


def test_detect_spikes_handles_constant_series():
    frame = pd.DataFrame(
        {"date": pd.date_range("2011-01-29", periods=10, freq="D"), "units": [5] * 10}
    )
    assert detect_spikes(frame)["spike_days"] == []


def test_event_effect_uses_is_event_flag(tiny_sales, tiny_calendar):
    from src.data.pipeline import attach_calendar_fields, melt_sales_long

    day_cols = [f"d_{index}" for index in range(1, N_DAYS + 1)]
    from src.data.pipeline import clean_calendar

    panel = attach_calendar_fields(
        melt_sales_long(tiny_sales, day_cols), clean_calendar(tiny_calendar)
    )
    effect = event_effect(panel)

    assert set(effect["cat_id"]) == {"FOODS", "HOBBIES"}
    assert effect["mean_units_event"].notna().all()


def test_demand_distribution_zero_share():
    panel = pd.DataFrame(
        {
            "id": ["a"] * 4 + ["b"] * 4,
            "units": [0, 1, 0, 1, 2, 2, 2, 2],
        }
    )
    stats = demand_distribution(panel)

    assert stats["series"] == 2
    assert stats["share_zero_day_rows_percent"] == 25.0


def test_compute_eda_summary_is_json_safe(tiny_sales, tiny_calendar, tiny_prices):
    from src.data.pipeline import (
        attach_calendar_fields,
        attach_prices,
        clean_calendar,
        compute_aggregates,
        melt_sales_long,
    )

    day_cols = [f"d_{index}" for index in range(1, N_DAYS + 1)]
    calendar = clean_calendar(tiny_calendar)
    aggregates = compute_aggregates(tiny_sales, calendar, day_cols)
    panel = attach_prices(
        attach_calendar_fields(melt_sales_long(tiny_sales, day_cols), calendar),
        tiny_prices,
    )
    price_stats = pd.DataFrame(
        {
            "store_id": ["CA_1"],
            "item_id": ["FOODS_3_090"],
            "mean_price": [3.0],
            "min_price": [2.5],
            "max_price": [3.0],
            "price_changes": [4],
        }
    )

    summary = compute_eda_summary(
        aggregates["aggregates/daily_total_units"],
        aggregates["aggregates/daily_store_units"],
        aggregates["aggregates/daily_category_units"],
        panel,
        price_stats,
        calendar,
    )

    json.dumps(summary["temporal"])  # must not raise
    assert summary["insights"]
    assert summary["demand"]["series"] == 3


def test_run_eda_end_to_end(raw_dir: Path, tmp_path: Path):
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    charts = tmp_path / "charts"

    run_pipeline(sample_series=3, raw_dir=raw_dir, out_dir=processed)
    run_eda(processed_dir=processed, reports_dir=reports, charts_dir=charts)

    output = reports / "eda_summary.json"
    assert output.is_file()

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["temporal"]["days"] == N_DAYS
    assert len(saved["charts"]) == 10

    for chart_path in saved["charts"]:
        assert Path(chart_path).is_file()
