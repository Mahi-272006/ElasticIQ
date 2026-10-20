"""Exploratory data analysis on the processed ElasticIQ datasets.

All functions are pure (DataFrame in, summaries out) so they can be tested
with fixtures. The orchestrator :func:`run_eda` loads processed data,
computes every summary, and returns a single JSON-serializable report that
the API and dashboard reuse.

Interpretation notes (``insights``) explain what each finding means for
forecasting or retail decisions; they describe patterns in the data, not
causal effects.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPIKE_ZSCORE_THRESHOLD = 4.0


def temporal_summary(daily_total: pd.DataFrame) -> dict:
    """Daily, weekly, and monthly sales summary from total daily units."""

    frame = daily_total.sort_values("date").copy()
    frame["date"] = pd.to_datetime(frame["date"])

    weekly = frame.set_index("date")["units"].resample("W").sum()
    monthly = frame.set_index("date")["units"].resample("ME").sum()

    return {
        "days": int(len(frame)),
        "start_date": frame["date"].min().strftime("%Y-%m-%d"),
        "end_date": frame["date"].max().strftime("%Y-%m-%d"),
        "total_units": int(frame["units"].sum()),
        "daily_mean": round(float(frame["units"].mean()), 2),
        "daily_min": int(frame["units"].min()),
        "daily_max": int(frame["units"].max()),
        "weekly_mean": round(float(weekly.mean()), 2),
        "monthly_mean": round(float(monthly.mean()), 2),
    }


def day_of_week_pattern(
    daily_total: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """Mean total units by weekday (1 = Saturday in the M5 calendar)."""

    frame = daily_total.merge(
        calendar[["date", "weekday", "wday"]], on="date", how="left"
    )
    grouped = frame.groupby(["wday", "weekday"], sort=True)["units"]
    result = grouped.agg(["mean", "median", "std"]).reset_index()
    result["mean"] = result["mean"].round(2)
    return result


def monthly_seasonality(daily_total: pd.DataFrame) -> pd.DataFrame:
    """Mean total units by calendar month across all years."""

    frame = daily_total.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.month

    result = frame.groupby("month")["units"].mean().round(2).reset_index()
    result.columns = ["month", "mean_units"]
    return result


def store_category_comparison(
    daily_store: pd.DataFrame, daily_category: pd.DataFrame
) -> dict:
    """Totals and shares by store and category."""

    def _summarize(frame: pd.DataFrame, key: str) -> list[dict]:
        totals = frame.groupby(key)["units"].sum().sort_values(ascending=False)
        grand = float(totals.sum())
        return [
            {
                key: str(name),
                "units": int(value),
                "share_percent": round(value / grand * 100, 2) if grand else 0.0,
            }
            for name, value in totals.items()
        ]

    return {
        "stores": _summarize(daily_store, "store_id"),
        "categories": _summarize(daily_category, "cat_id"),
    }


def demand_distribution(panel: pd.DataFrame) -> dict:
    """Per-series demand levels and variability (sample panel)."""

    grouped = panel.groupby("id", sort=False)["units"]
    totals = grouped.sum()
    means = grouped.mean()
    stds = grouped.std().fillna(0.0)

    coefficient_of_variation = (stds / means.replace(0, np.nan)).dropna()

    return {
        "series": int(panel["id"].nunique()),
        "median_series_total_units": round(float(totals.median()), 2),
        "p10_series_total_units": round(float(totals.quantile(0.10)), 2),
        "p90_series_total_units": round(float(totals.quantile(0.90)), 2),
        "median_series_daily_mean": round(float(means.median()), 3),
        "median_cv": round(float(coefficient_of_variation.median()), 3)
        if len(coefficient_of_variation)
        else None,
        "share_zero_day_rows_percent": round(
            float((panel["units"] == 0).mean() * 100), 2
        ),
    }


def intermittency_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Fraction of zero-sales days per series, bucketed by demand level."""

    stats = panel.groupby("id", sort=False).agg(
        total_units=("units", "sum"),
        zero_share=("units", lambda values: float((values == 0).mean())),
        days=("units", "size"),
    )

    stats["demand_level"] = pd.cut(
        stats["total_units"],
        bins=[-0.5, 100, 1000, 10_000, np.inf],
        labels=["very_low", "low", "medium", "high"],
    )

    result = (
        stats.groupby("demand_level", observed=True)["zero_share"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    result["mean"] = (result["mean"] * 100).round(2)
    result["median"] = (result["median"] * 100).round(2)
    result.columns = [
        "demand_level",
        "mean_zero_share_percent",
        "median_zero_share_percent",
        "series",
    ]
    return result


def price_profile(price_stats: pd.DataFrame) -> dict:
    """Distribution of per-series price levels and price changes."""

    return {
        "series_priced": int(len(price_stats)),
        "median_price": round(float(price_stats["mean_price"].median()), 4),
        "min_price": round(float(price_stats["min_price"].min()), 4),
        "max_price": round(float(price_stats["max_price"].max()), 4),
        "median_price_range_ratio": round(
            float((price_stats["max_price"] / price_stats["min_price"]).median()), 3
        ),
        "series_with_any_price_change": int((price_stats["price_changes"] > 0).sum()),
        "median_price_changes": round(float(price_stats["price_changes"].median()), 1),
    }


def event_effect(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean units on event vs non-event days, by category (association only)."""

    grouped = panel.groupby(["cat_id", "is_event"])["units"].mean().round(3)
    result = grouped.unstack("is_event")
    result.columns = ["mean_units_non_event", "mean_units_event"]
    result["event_to_non_event_ratio"] = (
        result["mean_units_event"] / result["mean_units_non_event"]
    ).round(3)
    return result.reset_index()


def detect_spikes(daily_total: pd.DataFrame) -> dict:
    """Flag unusually high aggregate-sales days with a z-score rule."""

    frame = daily_total.sort_values("date").copy()
    mean = frame["units"].mean()
    std = frame["units"].std()

    if std == 0 or np.isnan(std):
        return {"threshold_zscore": SPIKE_ZSCORE_THRESHOLD, "spike_days": []}

    frame["zscore"] = (frame["units"] - mean) / std
    spikes = frame[frame["zscore"] >= SPIKE_ZSCORE_THRESHOLD].sort_values(
        "zscore", ascending=False
    )

    return {
        "threshold_zscore": SPIKE_ZSCORE_THRESHOLD,
        "spike_days": [
            {
                "date": str(row["date"])[:10],
                "units": int(row["units"]),
                "zscore": round(float(row["zscore"]), 2),
            }
            for _, row in spikes.head(20).iterrows()
        ],
    }


def build_insights(summary: dict) -> list[dict]:
    """Human-readable, data-grounded observations for the report."""

    temporal = summary["temporal"]
    dow = summary["day_of_week"]
    events = summary["events"]
    intermittency = summary["intermittency"]

    busiest = dow.loc[dow["mean"].idxmax()]
    quietest = dow.loc[dow["mean"].idxmin()]
    zero_share = summary["demand"]["share_zero_day_rows_percent"]
    event_ratio = events.iloc[0]["event_to_non_event_ratio"] if len(events) else "n/a"
    zero_min = intermittency["mean_zero_share_percent"].min()
    zero_max = intermittency["mean_zero_share_percent"].max()

    insights = [
        {
            "topic": "seasonality",
            "insight": (
                f"Busiest weekday is {busiest['weekday']} at {busiest['mean']:,.0f} "
                f"mean units/day vs {quietest['mean']:,.0f} on {quietest['weekday']}; "
                "day-of-week seasonality is strong enough that weekly seasonal "
                "baselines should beat flat baselines on weekly-aggregated targets."
            ),
        },
        {
            "topic": "intermittency",
            "insight": (
                f"{temporal['days']:,} days of history; {zero_share}% of series-day "
                "rows have zero sales. Intermittent, zero-inflated demand means "
                "WAPE-style aggregate metrics matter more than per-series "
                "percentage errors."
            ),
        },
        {
            "topic": "events",
            "insight": (
                "Event-day means differ from non-event days in the panel "
                f"(ratio {event_ratio} for the first category). This is an "
                "observational association: event periods also carry promotions "
                "and price changes, so it must not be read as a causal effect."
            ),
        },
        {
            "topic": "demand_levels",
            "insight": (
                "Zero-sales share varies by demand level "
                f"(mean {zero_min}%-{zero_max}% across buckets). Models should "
                "be evaluated per demand-frequency bucket, not just in aggregate."
            ),
        },
    ]
    return insights


def compute_eda_summary(
    daily_total: pd.DataFrame,
    daily_store: pd.DataFrame,
    daily_category: pd.DataFrame,
    panel: pd.DataFrame,
    price_stats: pd.DataFrame,
    calendar: pd.DataFrame,
) -> dict:
    """Compute the full EDA summary from processed datasets."""

    summary = {
        "temporal": temporal_summary(daily_total),
        "day_of_week": day_of_week_pattern(daily_total, calendar),
        "monthly_seasonality": monthly_seasonality(daily_total),
        "store_category": store_category_comparison(daily_store, daily_category),
        "demand": demand_distribution(panel),
        "intermittency": intermittency_profile(panel),
        "prices": price_profile(price_stats),
        "events": event_effect(panel),
        "spikes": detect_spikes(daily_total),
    }
    summary["insights"] = build_insights(summary)
    return summary
