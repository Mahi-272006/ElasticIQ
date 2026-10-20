"""Deterministic preprocessing from the raw M5 CSVs to processed datasets.

The raw files are never modified. The pipeline produces typed, compact
datasets under ``data/processed`` that the forecasting, pricing, inventory,
and API layers consume:

- ``calendar``                     cleaned calendar with event flags
- ``products``                     unique product-store metadata
- ``aggregates/daily_total_units`` total daily units across all series
- ``aggregates/daily_category_units`` daily units by category
- ``aggregates/daily_store_units`` daily units by store
- ``sample_panel``                 long panel (units + price) for the first
                                   N series in file order (deterministic)
- ``series_history_tail``          last 56 days, wide, for ALL series
                                   (enables lag-based inference anywhere)
- ``price_stats``                  per store-item price history summary

Run with::

    python -m src.data.pipeline --sample-series 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.loader import RAW_DATA_DIR, day_columns, load_csv
from src.data.storage import processed_path, read_frame, write_frame

logger = logging.getLogger(__name__)

METADATA_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
HISTORY_TAIL_DAYS = 56  # 28-day lags + 28-day rolling context at inference time

PRICE_CHUNK_SIZE = 500_000


# ----------------------------------------------------------- pure helpers ---


def clean_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Return a typed calendar with parsed dates and event flags."""

    frame = calendar.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["day_idx"] = frame["d"].str[2:].astype(int)

    event_names = frame[["event_name_1", "event_name_2"]].astype("string")
    frame["is_event"] = event_names.notna().any(axis=1).astype(np.int8)

    for column in ["wday", "month", "year", "wm_yr_wk"]:
        frame[column] = frame[column].astype(np.int16)

    for column in ["snap_CA", "snap_TX", "snap_WI"]:
        if column in frame.columns:
            frame[column] = frame[column].astype(np.int8)

    return frame


def check_calendar_sales_mapping(
    calendar: pd.DataFrame, sales_day_columns: list[str]
) -> dict:
    """Verify that ``d_N`` sales columns map 1:1 onto calendar rows."""

    calendar_days = set(calendar["d"])
    sales_days = set(sales_day_columns)

    missing_in_calendar = sorted(sales_days - calendar_days)
    numbers = sorted(int(col[2:]) for col in sales_day_columns)
    consecutive = numbers == list(range(1, len(numbers) + 1))

    by_day = calendar.set_index("d")
    first_day = sales_day_columns[0]
    last_day = sales_day_columns[-1]

    def _date_str(day: str) -> str | None:
        if day not in by_day.index:
            return None
        return pd.Timestamp(by_day.loc[day, "date"]).strftime("%Y-%m-%d")

    return {
        "sales_day_columns": len(sales_day_columns),
        "calendar_rows": int(len(calendar)),
        "sales_days_missing_from_calendar": missing_in_calendar,
        "day_columns_consecutive_from_d_1": consecutive,
        "first_sales_day": first_day,
        "last_sales_day": last_day,
        "first_date": _date_str(first_day),
        "last_date": _date_str(last_day),
        "passed": not missing_in_calendar and consecutive,
    }


def melt_sales_long(sales: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    """Melt wide sales rows to a long frame: one row per series per day."""

    long = pd.melt(
        sales,
        id_vars=METADATA_COLUMNS,
        value_vars=day_cols,
        var_name="d",
        value_name="units",
    )
    long["day_idx"] = long["d"].str[2:].astype(np.int32)
    long.sort_values(["id", "day_idx"], inplace=True, ignore_index=True)
    return long


def attach_calendar_fields(long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Add date and week columns to a long sales frame via the ``d`` mapping.

    Expects a calendar cleaned by :func:`clean_calendar` (which provides the
    ``is_event`` flag).
    """

    fields = calendar[["d", "date", "wm_yr_wk", "wday", "month", "is_event"]]
    return long.merge(fields, on="d", how="left", validate="many_to_one")


def attach_prices(long: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join weekly sell prices onto daily rows.

    Items not yet on sale have no price row; the join leaves ``sell_price``
    as NaN, which is legitimate (product not yet introduced).
    """

    return long.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )


def compute_aggregates(
    sales: pd.DataFrame, calendar: pd.DataFrame, day_cols: list[str]
) -> dict[str, pd.DataFrame]:
    """Compute total, category, and store daily unit aggregates."""

    date_map = calendar.set_index("d")["date"]

    def _grouped_to_frame(wide: pd.DataFrame, key: str) -> pd.DataFrame:
        stacked = wide.stack()
        frame = stacked.rename("units").reset_index()
        frame.columns = [key, "d", "units"]
        frame["date"] = frame["d"].map(date_map)
        return frame

    total = sales[day_cols].sum(axis=0)
    total_frame = pd.DataFrame({"d": total.index, "units": total.to_numpy()})
    total_frame["date"] = total_frame["d"].map(date_map)

    by_category = _grouped_to_frame(
        sales.groupby("cat_id", sort=True)[day_cols].sum(), "cat_id"
    )
    by_store = _grouped_to_frame(
        sales.groupby("store_id", sort=True)[day_cols].sum(), "store_id"
    )

    # Deterministic date ordering for downstream consumers.
    total_frame.sort_values("date", inplace=True, ignore_index=True)
    by_category.sort_values(["cat_id", "date"], inplace=True, ignore_index=True)
    by_store.sort_values(["store_id", "date"], inplace=True, ignore_index=True)

    return {
        "aggregates/daily_total_units": total_frame,
        "aggregates/daily_category_units": by_category,
        "aggregates/daily_store_units": by_store,
    }


def summarize_prices_chunked(
    price_path: Path, chunk_size: int = PRICE_CHUNK_SIZE
) -> tuple[pd.DataFrame, dict]:
    """Summarize per store-item price history, reading the CSV in chunks."""

    running: dict[tuple[str, str], dict] = {}
    last_price: dict[tuple[str, str], float] = {}
    unmatched_weeks: set[int] = set()
    rows = 0

    for chunk in pd.read_csv(price_path, chunksize=chunk_size):
        rows += len(chunk)
        chunk = chunk.sort_values(["store_id", "item_id", "wm_yr_wk"])

        grouped = chunk.groupby(["store_id", "item_id"], sort=False)
        for (store_id, item_id), group in grouped:
            key = (store_id, item_id)
            prices = group["sell_price"].to_numpy(dtype=np.float64)
            weeks = group["wm_yr_wk"].to_numpy()

            entry = running.setdefault(
                key,
                {
                    "weeks": 0,
                    "sum": 0.0,
                    "sumsq": 0.0,
                    "min": np.inf,
                    "max": -np.inf,
                    "first_price": float(prices[0]),
                    "last_price": float(prices[-1]),
                    "price_changes": 0,
                    "first_wm_yr_wk": int(weeks[0]),
                    "last_wm_yr_wk": int(weeks[-1]),
                },
            )

            entry["weeks"] += len(prices)
            entry["sum"] += float(prices.sum())
            entry["sumsq"] += float((prices**2).sum())
            entry["min"] = min(entry["min"], float(prices.min()))
            entry["max"] = max(entry["max"], float(prices.max()))

            # Count changes against the previous week, including across
            # chunk boundaries (dicts persist between chunks).
            previous = last_price.get(key)
            for price in prices:
                if previous is not None and price != previous:
                    entry["price_changes"] += 1
                previous = price
            last_price[key] = previous

        unmatched_weeks.update(chunk["wm_yr_wk"].unique().tolist())

    records = []
    for (store_id, item_id), entry in running.items():
        count = entry["weeks"]
        mean = entry["sum"] / count
        variance = max(entry["sumsq"] / count - mean**2, 0.0)
        records.append(
            {
                "store_id": store_id,
                "item_id": item_id,
                "weeks_priced": count,
                "first_wm_yr_wk": entry["first_wm_yr_wk"],
                "last_wm_yr_wk": entry["last_wm_yr_wk"],
                "first_price": round(entry["first_price"], 4),
                "last_price": round(entry["last_price"], 4),
                "min_price": round(entry["min"], 4),
                "max_price": round(entry["max"], 4),
                "mean_price": round(mean, 4),
                "std_price": round(variance**0.5, 4),
                "price_changes": entry["price_changes"],
            }
        )

    stats = pd.DataFrame(records)
    stats.sort_values(["store_id", "item_id"], ignore_index=True, inplace=True)
    report = {
        "price_rows": rows,
        "store_item_series": len(stats),
        "weeks_seen_in_prices": sorted(int(week) for week in unmatched_weeks),
    }
    return stats, report


# ----------------------------------------------------------- orchestrator ---


def run_pipeline(
    sample_series: int | None = None,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    history_tail_days: int = HISTORY_TAIL_DAYS,
) -> dict:
    """Run the full deterministic preprocessing pipeline."""

    settings = get_settings()
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    out_dir = Path(out_dir) if out_dir is not None else settings.processed_data_dir
    sample_series = sample_series or settings.sample_series

    started = time.perf_counter()
    outputs: dict[str, str] = {}

    # 1. Calendar ------------------------------------------------------------
    logger.info("Cleaning calendar...")
    calendar = clean_calendar(load_csv("calendar.csv", data_dir=raw_dir))
    outputs["calendar"] = str(write_frame(calendar, out_dir / "calendar"))

    # 2. Sales (wide, compact dtypes) ----------------------------------------
    logger.info("Loading sales_train_validation.csv with compact dtypes...")
    header = load_csv("sales_train_validation.csv", nrows=0, data_dir=raw_dir)
    sales_day_columns = day_columns(header.columns)
    sales = load_csv(
        "sales_train_validation.csv",
        data_dir=raw_dir,
        dtype={col: np.int16 for col in sales_day_columns},
    )

    mapping = check_calendar_sales_mapping(calendar, sales_day_columns)
    if not mapping["passed"]:
        raise ValueError(f"Calendar/sales day mapping failed: {mapping}")
    logger.info(
        "Sales coverage: %s (%s) to %s (%s)",
        mapping["first_sales_day"],
        mapping["first_date"],
        mapping["last_sales_day"],
        mapping["last_date"],
    )

    # 3. Product metadata ----------------------------------------------------
    products = sales[METADATA_COLUMNS].copy()
    outputs["products"] = str(write_frame(products, out_dir / "products"))

    # 4. Aggregates ----------------------------------------------------------
    logger.info("Computing daily aggregates...")
    for name, frame in compute_aggregates(sales, calendar, sales_day_columns).items():
        outputs[name] = str(write_frame(frame, out_dir / name))

    # 5. Sample long panel with prices ---------------------------------------
    selected = min(sample_series, len(sales))
    logger.info("Building long panel for the first %s series...", f"{selected:,}")
    sample = sales.head(selected)
    panel = melt_sales_long(sample, sales_day_columns)
    panel = attach_calendar_fields(panel, calendar)

    prices = load_csv(
        "sell_prices.csv",
        data_dir=raw_dir,
        dtype={"store_id": "category", "item_id": "category"},
    )
    panel = attach_prices(panel, prices)
    outputs["sample_panel"] = str(write_frame(panel, out_dir / "sample_panel"))

    # 6. History tail for ALL series -----------------------------------------
    tail_columns = sales_day_columns[-min(history_tail_days, len(sales_day_columns)) :]
    history_tail = sales[METADATA_COLUMNS + tail_columns].copy()
    outputs["series_history_tail"] = str(
        write_frame(history_tail, out_dir / "series_history_tail")
    )

    # 7. Price statistics ----------------------------------------------------
    logger.info("Summarizing sell_prices.csv in chunks...")
    price_stats, price_report = summarize_prices_chunked(raw_dir / "sell_prices.csv")

    calendar_weeks = set(calendar["wm_yr_wk"].tolist())
    unmatched_price_weeks = sorted(
        week
        for week in price_report["weeks_seen_in_prices"]
        if week not in calendar_weeks
    )
    price_report["price_weeks_missing_from_calendar"] = unmatched_price_weeks
    del price_report["weeks_seen_in_prices"]

    # Price keys that do not match any known product.
    known_pairs = set(zip(products["store_id"], products["item_id"], strict=True))
    stat_pairs = set(zip(price_stats["store_id"], price_stats["item_id"], strict=True))
    orphan_price_keys = sorted(stat_pairs - known_pairs)
    price_report["price_keys_without_product"] = len(orphan_price_keys)

    outputs["price_stats"] = str(write_frame(price_stats, out_dir / "price_stats"))

    # 8. Report ----------------------------------------------------------------
    report = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "sample_selection": (
            f"first {selected:,} series in sales_train_validation.csv file order "
            "(deterministic head selection)"
        ),
        "sample_series": selected,
        "total_series": int(len(sales)),
        "calendar_mapping": mapping,
        "price_report": price_report,
        "panel_rows": int(len(panel)),
        "history_tail_days": len(tail_columns),
        "outputs": outputs,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "storage_format": (
            "parquet" if outputs["calendar"].endswith(".parquet") else "csv.gz"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Pipeline finished in %s s; report at %s",
        report["elapsed_seconds"],
        report_path,
    )

    return report


def load_processed(name: str, out_dir: Path | None = None) -> pd.DataFrame:
    """Load a processed dataset by name from the processed directory."""

    if out_dir is None:
        out_dir = get_settings().processed_data_dir
    else:
        out_dir = Path(out_dir)
    path = processed_path(out_dir, name)
    if path is None:
        raise FileNotFoundError(
            f"Processed dataset '{name}' not found in {out_dir}. "
            "Run `python -m src.data.pipeline` first."
        )
    return read_frame(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ElasticIQ processed datasets")
    parser.add_argument(
        "--sample-series",
        type=int,
        default=None,
        help="Number of series for the long panel (default: ELASTICIQ_SAMPLE_SERIES)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    report = run_pipeline(sample_series=args.sample_series)

    print("\n" + "=" * 60)
    print("ELASTICIQ - DATA PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total series: {report['total_series']:,}")
    print(f"Panel series: {report['sample_series']:,} ({report['panel_rows']:,} rows)")
    print(
        f"Coverage: {report['calendar_mapping']['first_date']} "
        f"to {report['calendar_mapping']['last_date']}"
    )
    print(f"Storage: {report['storage_format']}")
    print(f"Elapsed: {report['elapsed_seconds']} s")


if __name__ == "__main__":
    main()
