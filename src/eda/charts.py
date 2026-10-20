"""Saved analytical charts for the EDA report.

Every function takes plain DataFrames and a target directory, renders one
chart with matplotlib's non-interactive backend, and returns the saved path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 150


def _save(fig: plt.Figure, target_dir: Path, name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def chart_daily_weekly_trend(daily_total: pd.DataFrame, target_dir: Path) -> Path:
    """Daily units with a weekly rolling mean overlay."""

    frame = daily_total.sort_values("date").copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["weekly_mean"] = frame["units"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(frame["date"], frame["units"], linewidth=0.7, label="Daily units")
    ax.plot(frame["date"], frame["weekly_mean"], linewidth=1.6, label="7-day mean")
    ax.set_title("ElasticIQ: Total Daily Units Sold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend(loc="upper right")

    return _save(fig, target_dir, "daily_weekly_trend.png")


def chart_day_of_week(dow: pd.DataFrame, target_dir: Path) -> Path:
    """Mean units by weekday."""

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(dow["weekday"], dow["mean"], color="#2a6fbb")
    ax.set_title("Mean Daily Units by Weekday")
    ax.set_ylabel("Mean units")
    ax.tick_params(axis="x", rotation=30)

    return _save(fig, target_dir, "day_of_week_pattern.png")


def chart_monthly_seasonality(monthly: pd.DataFrame, target_dir: Path) -> Path:
    """Mean units by calendar month across years."""

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(monthly["month"], monthly["mean_units"], marker="o", linewidth=1.5)
    ax.set_xticks(sorted(monthly["month"]))
    ax.set_title("Seasonality: Mean Units by Calendar Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean units")

    return _save(fig, target_dir, "monthly_seasonality.png")


def chart_store_comparison(stores: pd.DataFrame, target_dir: Path) -> Path:
    """Total units by store."""

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(stores["store_id"], stores["units"], color="#2a9d6f")
    ax.set_title("Total Units by Store")
    ax.set_ylabel("Units")
    ax.tick_params(axis="x", rotation=30)

    return _save(fig, target_dir, "store_comparison.png")


def chart_category_comparison(categories: pd.DataFrame, target_dir: Path) -> Path:
    """Total units and share by category."""

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(categories["cat_id"], categories["units"], color="#e07a1f")
    ax.set_title("Total Units by Category")
    ax.set_xlabel("Units")

    return _save(fig, target_dir, "category_comparison.png")


def chart_demand_distribution(series_totals: pd.Series, target_dir: Path) -> Path:
    """Distribution of per-series total units (log-scaled x-axis)."""

    fig, ax = plt.subplots(figsize=(9, 5))
    positive = series_totals[series_totals > 0]
    ax.hist(positive, bins=50, color="#6a4c93")
    ax.set_xscale("log")
    ax.set_title("Distribution of Series Total Units (log scale)")
    ax.set_xlabel("Total units per series")
    ax.set_ylabel("Number of series")

    return _save(fig, target_dir, "demand_distribution.png")


def chart_intermittency(intermittency: pd.DataFrame, target_dir: Path) -> Path:
    """Mean zero-sales share by demand level."""

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        intermittency["demand_level"].astype(str),
        intermittency["mean_zero_share_percent"],
        color="#457b9d",
    )
    ax.set_title("Mean Share of Zero-Sales Days by Demand Level")
    ax.set_ylabel("Zero-sales days (%)")

    return _save(fig, target_dir, "intermittency_by_demand_level.png")


def chart_price_distribution(price_stats: pd.DataFrame, target_dir: Path) -> Path:
    """Histogram of per-series mean prices."""

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(price_stats["mean_price"], bins=60, color="#bc4749")
    ax.set_title("Distribution of Series Mean Sell Price")
    ax.set_xlabel("Mean price")
    ax.set_ylabel("Number of series")

    return _save(fig, target_dir, "price_distribution.png")


def chart_event_effect(events: pd.DataFrame, target_dir: Path) -> Path:
    """Event vs non-event mean units per category (association only)."""

    melted = events.melt(
        id_vars="cat_id",
        value_vars=["mean_units_non_event", "mean_units_event"],
        var_name="period",
        value_name="mean_units",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for period, group in melted.groupby("period"):
        label = (
            "Event days"
            if "event" in period and "non" not in period
            else "Non-event days"
        )
        ax.bar(
            [f"{cat}\n{label}" for cat in group["cat_id"]],
            group["mean_units"],
            label=label,
        )
    ax.set_title("Mean Units on Event vs Non-Event Days (association, not causation)")
    ax.set_ylabel("Mean units per series-day")
    ax.legend()

    return _save(fig, target_dir, "event_effect.png")


def chart_spikes(
    daily_total: pd.DataFrame, spikes: list[dict], target_dir: Path
) -> Path:
    """Daily totals with flagged spike days marked."""

    frame = daily_total.sort_values("date").copy()
    frame["date"] = pd.to_datetime(frame["date"])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(frame["date"], frame["units"], linewidth=0.7, color="#555555")

    spike_dates = [pd.Timestamp(spike["date"]) for spike in spikes]
    if spike_dates:
        spike_frame = frame[frame["date"].isin(spike_dates)]
        ax.scatter(
            spike_frame["date"],
            spike_frame["units"],
            color="red",
            s=28,
            zorder=3,
            label="Spike day (z >= 4)",
        )
        ax.legend(loc="upper right")

    ax.set_title("Daily Units with Detected Demand Spikes")
    ax.set_ylabel("Units")

    return _save(fig, target_dir, "demand_spikes.png")
