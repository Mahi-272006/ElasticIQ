"""Data-quality validation for the raw M5 files.

The module has two layers:

1. Pure DataFrame validators (``validate_calendar_frame`` and friends) that
   return a list of check dictionaries. These are unit-testable with small
   fixtures and never touch the filesystem.
2. File-based wrappers (``validate_dataset_sample``, ``run_sample_validation``)
   that load a sample from disk and delegate to the pure validators.

Every check is a dictionary with at least ``check`` (name) and ``passed``
(bool) keys, so results can be rendered or serialized as structured reports.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import EXPECTED_FILES, day_columns, load_csv

SALES_FILES = ("sales_train_validation.csv", "sales_train_evaluation.csv")

REQUIRED_COLUMNS = {
    "calendar.csv": {"date", "wm_yr_wk", "weekday", "wday", "month", "year", "d"},
    "sales_train_validation.csv": {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    },
    "sales_train_evaluation.csv": {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    },
    "sell_prices.csv": {"store_id", "item_id", "wm_yr_wk", "sell_price"},
    "sample_submission.csv": {"id"},
}

METADATA_COLUMNS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")


def _check(name: str, passed: bool, **details) -> dict:
    return {"check": name, "passed": bool(passed), **details}


def validate_required_columns(file_name: str, columns) -> dict:
    """Check that required columns exist for a known file."""

    if file_name not in REQUIRED_COLUMNS:
        raise KeyError(f"No required-column definition for: {file_name}")

    missing = sorted(REQUIRED_COLUMNS[file_name] - set(columns))

    return _check("required_columns", len(missing) == 0, missing_columns=missing)


def validate_calendar_frame(df: pd.DataFrame) -> list[dict]:
    """Validate a calendar DataFrame (pure; no file access)."""

    results = [validate_required_columns("calendar.csv", df.columns)]
    if not results[0]["passed"]:
        return results

    dates = pd.to_datetime(df["date"], errors="coerce")

    results.extend(
        [
            _check("calendar_dates_parseable", dates.notna().all()),
            _check("calendar_dates_unique", not df["date"].duplicated().any()),
            _check("calendar_day_ids_unique", not df["d"].duplicated().any()),
        ]
    )

    # The d_N identifiers must be consecutive integers starting at d_1.
    day_ids = df["d"]
    numeric_part = day_ids.str.extract(r"^d_(\d+)$", expand=False)
    results.append(
        _check(
            "calendar_day_ids_well_formed",
            numeric_part.notna().all(),
            invalid_day_ids=int(numeric_part.isna().sum()),
        )
    )

    if numeric_part.notna().all():
        numbers = sorted(numeric_part.astype(int).tolist())
        consecutive = numbers == list(range(numbers[0], numbers[0] + len(numbers)))
        results.append(_check("calendar_day_ids_consecutive", consecutive))

    # Dates must increase monotonically alongside d_N.
    parsed = dates.dropna()
    results.append(
        _check("calendar_dates_monotonic_increasing", parsed.is_monotonic_increasing)
    )

    return results


def validate_sales_frame(df: pd.DataFrame, file_name: str) -> list[dict]:
    """Validate a wide sales DataFrame (pure; no file access)."""

    results = [validate_required_columns(file_name, df.columns)]
    if not results[0]["passed"]:
        return results

    columns = day_columns(df.columns)
    results.append(
        _check(
            "sales_day_columns_present",
            len(columns) > 0,
            day_column_count=len(columns),
        )
    )
    results.append(_check("sales_ids_unique", not df["id"].duplicated().any()))

    if not columns:
        return results

    sales = df[columns].apply(pd.to_numeric, errors="coerce")

    # M5 ids look like "{item_id}_{store_id}[_evaluation]".
    id_prefix_ok = all(
        str(identifier).startswith(str(item))
        for identifier, item in zip(df["id"], df["item_id"], strict=False)
    )

    results.extend(
        [
            _check("sales_values_numeric", sales.notna().all().all()),
            _check("sales_values_non_negative", sales.ge(0).all().all()),
            _check("sales_ids_match_item_prefix", id_prefix_ok),
        ]
    )

    return results


def validate_prices_frame(df: pd.DataFrame) -> list[dict]:
    """Validate a sell-prices DataFrame (pure; no file access)."""

    results = [validate_required_columns("sell_prices.csv", df.columns)]
    if not results[0]["passed"]:
        return results

    prices = pd.to_numeric(df["sell_price"], errors="coerce")

    results.extend(
        [
            _check("prices_numeric", prices.notna().all()),
            _check("prices_positive", prices.gt(0).all()),
            _check(
                "price_keys_unique",
                not df.duplicated(["store_id", "item_id", "wm_yr_wk"]).any(),
            ),
        ]
    )

    return results


def validate_submission_frame(df: pd.DataFrame) -> list[dict]:
    """Validate a sample-submission DataFrame (pure; no file access)."""

    results = [validate_required_columns("sample_submission.csv", df.columns)]
    if not results[0]["passed"]:
        return results

    results.append(_check("submission_ids_unique", not df["id"].duplicated().any()))

    return results


def validate_dataset_sample(file_name: str, data_dir: Path | None = None) -> dict:
    """Run initial data-quality checks on up to 1,000 rows of a file."""

    if file_name not in EXPECTED_FILES:
        raise KeyError(f"Unknown dataset file: {file_name}")

    df = load_csv(file_name, nrows=1000, data_dir=data_dir)

    if file_name == "calendar.csv":
        results = validate_calendar_frame(df)
    elif file_name in SALES_FILES:
        results = validate_sales_frame(df, file_name)
    elif file_name == "sell_prices.csv":
        results = validate_prices_frame(df)
    else:
        results = validate_submission_frame(df)

    return {"file": file_name, "sample_rows": len(df), "results": results}


def run_sample_validation(data_dir: Path | None = None) -> list[dict]:
    """Validate a sample from each expected dataset."""

    return [
        validate_dataset_sample(file_name, data_dir=data_dir)
        for file_name in EXPECTED_FILES
    ]


def summarize_results(reports: list[dict]) -> dict:
    """Aggregate per-file check reports into a single summary."""

    total = 0
    failed = []

    for report in reports:
        for result in report["results"]:
            total += 1
            if not result["passed"]:
                failed.append({"file": report["file"], "check": result["check"]})

    return {
        "checks_run": total,
        "checks_failed": len(failed),
        "all_passed": len(failed) == 0,
        "failures": failed,
    }
