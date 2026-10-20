"""Run the full EDA: summary JSON + saved charts.

Usage::

    python -m src.eda.run

Outputs land in ``artifacts/reports/eda_summary.json`` and
``artifacts/reports/charts/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.data.pipeline import load_processed
from src.data.storage import processed_path, read_frame
from src.eda.analysis import compute_eda_summary
from src.eda.charts import (
    chart_category_comparison,
    chart_daily_weekly_trend,
    chart_day_of_week,
    chart_demand_distribution,
    chart_event_effect,
    chart_intermittency,
    chart_monthly_seasonality,
    chart_price_distribution,
    chart_spikes,
    chart_store_comparison,
)

logger = logging.getLogger(__name__)


def _serialize(summary: dict) -> dict:
    """Convert DataFrames inside the summary into JSON-safe records."""

    serialized = {}
    for key, value in summary.items():
        if isinstance(value, pd.DataFrame):
            frame = value.copy()
            for column in frame.columns:
                if str(frame[column].dtype) == "category":
                    frame[column] = frame[column].astype(str)
            serialized[key] = frame.to_dict(orient="records")
        else:
            serialized[key] = value
    return serialized


def run_eda(
    processed_dir: Path | None = None,
    reports_dir: Path | None = None,
    charts_dir: Path | None = None,
) -> dict:
    """Compute the EDA summary and save all charts and the JSON report."""

    settings = get_settings()
    processed_dir = (
        Path(processed_dir) if processed_dir else settings.processed_data_dir
    )
    reports_dir = Path(reports_dir) if reports_dir else settings.reports_dir
    charts_dir = Path(charts_dir) if charts_dir else settings.charts_dir

    logger.info("Loading processed datasets from %s ...", processed_dir)
    daily_total = load_processed("aggregates/daily_total_units", processed_dir)
    daily_store = load_processed("aggregates/daily_store_units", processed_dir)
    daily_category = load_processed("aggregates/daily_category_units", processed_dir)
    panel = load_processed("sample_panel", processed_dir)
    price_stats = load_processed("price_stats", processed_dir)
    calendar_path = processed_path(processed_dir, "calendar")
    if calendar_path is None:
        raise FileNotFoundError(
            f"Processed calendar not found in {processed_dir}. "
            "Run `python -m src.data.pipeline` first."
        )
    calendar = read_frame(calendar_path)

    summary = compute_eda_summary(
        daily_total, daily_store, daily_category, panel, price_stats, calendar
    )

    logger.info("Rendering charts to %s ...", charts_dir)
    saved_charts = [
        chart_daily_weekly_trend(daily_total, charts_dir),
        chart_day_of_week(summary["day_of_week"], charts_dir),
        chart_monthly_seasonality(summary["monthly_seasonality"], charts_dir),
        chart_store_comparison(
            pd.DataFrame(summary["store_category"]["stores"]), charts_dir
        ),
        chart_category_comparison(
            pd.DataFrame(summary["store_category"]["categories"]), charts_dir
        ),
        chart_demand_distribution(panel.groupby("id")["units"].sum(), charts_dir),
        chart_intermittency(summary["intermittency"], charts_dir),
        chart_price_distribution(price_stats, charts_dir),
        chart_event_effect(summary["events"], charts_dir),
        chart_spikes(daily_total, summary["spikes"]["spike_days"], charts_dir),
    ]

    report = _serialize(summary)
    report["charts"] = [str(path) for path in saved_charts]
    report["panel_series"] = int(panel["id"].nunique())
    report["panel_note"] = (
        "Panel-based statistics (demand distribution, intermittency, events) "
        "use the deterministic sample panel, not all 30,490 series."
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    output = reports_dir / "eda_summary.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("EDA summary written to %s", output)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ElasticIQ EDA")
    parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    report = run_eda()

    print("\n" + "=" * 60)
    print("ELASTICIQ - EDA COMPLETE")
    print("=" * 60)
    temporal = report["temporal"]
    print(f"Coverage: {temporal['start_date']} to {temporal['end_date']}")
    print(f"Total units: {temporal['total_units']:,}")
    print(f"Charts saved: {len(report['charts'])}")
    print("Report: artifacts/reports/eda_summary.json")


if __name__ == "__main__":
    main()
