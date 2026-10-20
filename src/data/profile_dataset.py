import json
from pathlib import Path

import pandas as pd

from src.data.loader import RAW_DATA_DIR, load_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"


def profile_sales() -> dict:
    """Summarize product-level sales without reshaping the full dataset."""

    sales = load_csv("sales_train_validation.csv")

    metadata_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_") and column[2:].isdigit()
    ]

    daily_sales = sales[day_columns].sum(axis=0)
    product_totals = sales[day_columns].sum(axis=1)

    store_sales = (
        sales.groupby("store_id")[day_columns]
        .sum()
        .sum(axis=1)
        .sort_values(ascending=False)
    )

    summary = pd.DataFrame(
        {
            "cat_id": sales["cat_id"].to_numpy(),
            "item_id": sales["item_id"].to_numpy(),
            "total_sales": product_totals.to_numpy(),
        }
    )

    category_sales = (
        summary.groupby("cat_id")["total_sales"].sum().sort_values(ascending=False)
    )

    item_sales = (
        summary.groupby("item_id")["total_sales"].sum().sort_values(ascending=False)
    )

    return {
        "product_rows": int(len(sales)),
        "metadata_columns": metadata_columns,
        "daily_observation_columns": int(len(day_columns)),
        "total_units_sold": int(product_totals.sum()),
        "zero_sales_products": int(product_totals.eq(0).sum()),
        "products_with_any_zero_sales_day": int(
            sales[day_columns].eq(0).any(axis=1).sum()
        ),
        "daily_sales": {
            "mean": round(float(daily_sales.mean()), 2),
            "median": round(float(daily_sales.median()), 2),
            "minimum": int(daily_sales.min()),
            "maximum": int(daily_sales.max()),
        },
        "top_stores_by_units": [
            {"store_id": str(index), "units": int(value)}
            for index, value in store_sales.head(10).items()
        ],
        "category_sales": [
            {"category": str(index), "units": int(value)}
            for index, value in category_sales.items()
        ],
        "top_10_items_by_units": [
            {"item_id": str(index), "units": int(value)}
            for index, value in item_sales.head(10).items()
        ],
    }


def profile_prices() -> dict:
    """Summarize prices using chunked reading to limit memory usage."""

    price_path = RAW_DATA_DIR / "sell_prices.csv"

    row_count = 0
    missing_prices = 0
    non_positive_prices = 0
    price_sum = 0.0
    minimum_price = float("inf")
    maximum_price = float("-inf")

    for chunk in pd.read_csv(price_path, chunksize=100_000):
        prices = pd.to_numeric(chunk["sell_price"], errors="coerce")
        valid = prices.dropna()

        row_count += len(chunk)
        missing_prices += int(prices.isna().sum())
        non_positive_prices += int(prices.le(0).sum())

        if not valid.empty:
            price_sum += float(valid.sum())
            minimum_price = min(minimum_price, float(valid.min()))
            maximum_price = max(maximum_price, float(valid.max()))

    valid_count = row_count - missing_prices

    return {
        "rows": row_count,
        "missing_prices": missing_prices,
        "non_positive_prices": non_positive_prices,
        "mean_price": round(price_sum / valid_count, 4) if valid_count else None,
        "minimum_price": minimum_price if valid_count else None,
        "maximum_price": maximum_price if valid_count else None,
    }


def profile_calendar() -> dict:
    """Summarize calendar coverage and event availability."""

    calendar = load_csv("calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")

    event_columns = [
        column
        for column in [
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
        ]
        if column in calendar.columns
    ]

    return {
        "rows": int(len(calendar)),
        "invalid_dates": int(calendar["date"].isna().sum()),
        "start_date": (
            calendar["date"].min().strftime("%Y-%m-%d")
            if calendar["date"].notna().any()
            else None
        ),
        "end_date": (
            calendar["date"].max().strftime("%Y-%m-%d")
            if calendar["date"].notna().any()
            else None
        ),
        "unique_dates": int(calendar["date"].nunique()),
        "event_columns": event_columns,
        "non_null_event_counts": {
            column: int(calendar[column].notna().sum()) for column in event_columns
        },
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Profiling sales data...")
    sales_report = profile_sales()

    print("Profiling price data...")
    price_report = profile_prices()

    print("Profiling calendar data...")
    calendar_report = profile_calendar()

    report = {
        "sales": sales_report,
        "prices": price_report,
        "calendar": calendar_report,
    }

    output_path = REPORT_DIR / "dataset_profile.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("ELASTICIQ - DATASET PROFILE")
    print("=" * 55)
    print(f"Products/series: {sales_report['product_rows']:,}")
    daily_obs = sales_report["daily_observation_columns"]
    print(f"Daily observations per series: {daily_obs:,}")
    print(f"Total units sold: {sales_report['total_units_sold']:,}")
    print(f"Products with zero total sales: {sales_report['zero_sales_products']:,}")
    start = calendar_report["start_date"]
    end = calendar_report["end_date"]
    print(f"Calendar dates: {start} to {end}")
    print(f"Price records: {price_report['rows']:,}")

    print("\nTop 5 stores by units sold:")
    for store in sales_report["top_stores_by_units"][:5]:
        print(f"  {store['store_id']}: {store['units']:,} units")

    print("\nSales by category:")
    for category in sales_report["category_sales"]:
        print(f"  {category['category']}: {category['units']:,} units")

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
