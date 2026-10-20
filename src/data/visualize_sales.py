from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save charts without opening GUI windows

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = PROJECT_ROOT / "artifacts" / "reports" / "charts"


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sales data...")
    sales = load_csv("sales_train_validation.csv")

    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_") and column[2:].isdigit()
    ]

    print("Loading calendar data...")
    calendar = load_csv("calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")

    # Match the sales columns to their corresponding calendar dates.
    calendar = calendar.set_index("d").reindex(day_columns)
    dates = calendar["date"]

    print("Calculating daily total sales...")
    daily_sales = sales[day_columns].sum(axis=0)
    daily_sales.index = dates.to_numpy()
    daily_sales.index.name = "date"
    daily_sales.name = "units_sold"

    # 1. Daily sales trend.
    plt.figure(figsize=(13, 5))
    plt.plot(daily_sales.index, daily_sales.values, linewidth=0.8)
    plt.title("ElasticIQ: Daily Retail Sales")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "daily_sales.png", dpi=150)
    plt.close()

    # 2. Weekly and monthly patterns.
    sales_series = daily_sales.sort_index()
    weekly_sales = sales_series.resample("W").sum()
    monthly_sales = sales_series.resample("MS").sum()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    axes[0].plot(weekly_sales.index, weekly_sales.values, linewidth=0.9)
    axes[0].set_title("Weekly Total Sales")
    axes[0].set_ylabel("Units sold")

    axes[1].plot(
        monthly_sales.index, monthly_sales.values, marker="o", markersize=2, linewidth=1
    )
    axes[1].set_title("Monthly Total Sales")
    axes[1].set_ylabel("Units sold")
    axes[1].set_xlabel("Month")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "weekly_monthly_sales.png", dpi=150)
    plt.close(fig)

    # 3. Historical units by category.
    product_totals = sales[day_columns].sum(axis=1)

    category_totals = (
        pd.DataFrame(
            {
                "cat_id": sales["cat_id"].to_numpy(),
                "_total": product_totals.to_numpy(),
            }
        )
        .groupby("cat_id")["_total"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    category_totals.sort_values().plot(kind="barh")
    plt.title("Total Units Sold by Category")
    plt.xlabel("Units sold")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "category_sales.png", dpi=150)
    plt.close()

    print("\nCharts saved successfully:")
    for filename in [
        "daily_sales.png",
        "weekly_monthly_sales.png",
        "category_sales.png",
    ]:
        print(f"  {CHART_DIR / filename}")


if __name__ == "__main__":
    main()
