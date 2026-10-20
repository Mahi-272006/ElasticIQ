"""Unit tests for data validation (pure validators + file wrappers)."""

from pathlib import Path

import pandas as pd

from src.data.validation import (
    REQUIRED_COLUMNS,
    run_sample_validation,
    summarize_results,
    validate_calendar_frame,
    validate_dataset_sample,
    validate_prices_frame,
    validate_required_columns,
    validate_sales_frame,
    validate_submission_frame,
)


def _passed(results: list[dict]) -> bool:
    return all(result["passed"] for result in results)


def _check_names(results: list[dict]) -> set[str]:
    return {result["check"] for result in results}


# ---------------------------------------------------------------- columns ---


def test_required_columns_pass():
    result = validate_required_columns("calendar.csv", REQUIRED_COLUMNS["calendar.csv"])
    assert result["passed"] is True
    assert result["missing_columns"] == []


def test_required_columns_fail_lists_missing():
    result = validate_required_columns("sell_prices.csv", {"store_id", "item_id"})
    assert result["passed"] is False
    assert result["missing_columns"] == ["sell_price", "wm_yr_wk"]


def test_unknown_file_raises():
    try:
        validate_required_columns("mystery.csv", set())
    except KeyError as exc:
        assert "mystery.csv" in str(exc)
    else:  # pragma: no cover - guard against silent pass
        raise AssertionError("Expected KeyError for unknown file")


# --------------------------------------------------------------- calendar ---


def test_calendar_frame_passes(tiny_calendar: pd.DataFrame):
    results = validate_calendar_frame(tiny_calendar)
    assert _passed(results)


def test_calendar_blank_events_are_legitimate(tiny_calendar: pd.DataFrame):
    # Most event cells are blank by design; this must not fail validation.
    results = validate_calendar_frame(tiny_calendar)
    assert _passed(results)


def test_calendar_duplicate_date_fails(tiny_calendar: pd.DataFrame):
    broken = tiny_calendar.copy()
    broken.loc[1, "date"] = broken.loc[0, "date"]

    results = validate_calendar_frame(broken)
    failed = {r["check"] for r in results if not r["passed"]}
    assert "calendar_dates_unique" in failed


def test_calendar_non_consecutive_day_ids_fail(tiny_calendar: pd.DataFrame):
    broken = tiny_calendar.copy()
    broken.loc[5, "d"] = "d_99"

    results = validate_calendar_frame(broken)
    failed = {r["check"] for r in results if not r["passed"]}
    assert "calendar_day_ids_consecutive" in failed


def test_calendar_unparseable_date_fails(tiny_calendar: pd.DataFrame):
    broken = tiny_calendar.copy()
    broken.loc[3, "date"] = "not-a-date"

    results = validate_calendar_frame(broken)
    failed = {r["check"] for r in results if not r["passed"]}
    assert "calendar_dates_parseable" in failed


# ------------------------------------------------------------------ sales ---


def test_sales_frame_passes(tiny_sales: pd.DataFrame):
    results = validate_sales_frame(tiny_sales, "sales_train_validation.csv")
    assert _passed(results)


def test_sales_negative_value_fails(tiny_sales: pd.DataFrame):
    broken = tiny_sales.copy()
    broken.loc[0, "d_1"] = -3

    results = validate_sales_frame(broken, "sales_train_validation.csv")
    failed = {r["check"] for r in results if not r["passed"]}
    assert "sales_values_non_negative" in failed


def test_sales_duplicate_id_fails(tiny_sales: pd.DataFrame):
    broken = tiny_sales.copy()
    broken.loc[1, "id"] = broken.loc[0, "id"]

    results = validate_sales_frame(broken, "sales_train_validation.csv")
    failed = {r["check"] for r in results if not r["passed"]}
    assert "sales_ids_unique" in failed


def test_sales_id_prefix_mismatch_fails(tiny_sales: pd.DataFrame):
    broken = tiny_sales.copy()
    broken.loc[0, "id"] = "WRONG_ITEM_CA_1"

    results = validate_sales_frame(broken, "sales_train_validation.csv")
    failed = {r["check"] for r in results if not r["passed"]}
    assert "sales_ids_match_item_prefix" in failed


# ----------------------------------------------------------------- prices ---


def test_prices_frame_passes(tiny_prices: pd.DataFrame):
    results = validate_prices_frame(tiny_prices)
    assert _passed(results)


def test_prices_non_positive_fails(tiny_prices: pd.DataFrame):
    broken = tiny_prices.copy()
    broken.loc[0, "sell_price"] = 0.0

    results = validate_prices_frame(broken)
    failed = {r["check"] for r in results if not r["passed"]}
    assert "prices_positive" in failed


def test_prices_duplicate_keys_fail(tiny_prices: pd.DataFrame):
    broken = pd.concat([tiny_prices, tiny_prices.iloc[[0]]], ignore_index=True)

    results = validate_prices_frame(broken)
    failed = {r["check"] for r in results if not r["passed"]}
    assert "price_keys_unique" in failed


# ------------------------------------------------------------ submission ---


def test_submission_frame_passes(tiny_submission: pd.DataFrame):
    results = validate_submission_frame(tiny_submission)
    assert _passed(results)


def test_submission_duplicate_ids_fail(tiny_submission: pd.DataFrame):
    broken = pd.concat([tiny_submission, tiny_submission.iloc[[0]]], ignore_index=True)

    results = validate_submission_frame(broken)
    failed = {r["check"] for r in results if not r["passed"]}
    assert "submission_ids_unique" in failed


# ------------------------------------------------------- file-level wrappers


def test_validate_dataset_sample_on_fixture_dir(raw_dir: Path):
    report = validate_dataset_sample("calendar.csv", data_dir=raw_dir)

    assert report["file"] == "calendar.csv"
    assert report["sample_rows"] > 0
    assert _passed(report["results"])


def test_run_sample_validation_covers_all_files(raw_dir: Path):
    reports = run_sample_validation(data_dir=raw_dir)

    assert len(reports) == len(REQUIRED_COLUMNS)
    summary = summarize_results(reports)

    assert summary["all_passed"] is True
    assert summary["checks_failed"] == 0
    assert summary["checks_run"] > 0


def test_summarize_results_reports_failures(tiny_prices: pd.DataFrame):
    broken = tiny_prices.copy()
    broken.loc[0, "sell_price"] = -1

    reports = [{"file": "sell_prices.csv", "results": validate_prices_frame(broken)}]
    summary = summarize_results(reports)

    assert summary["all_passed"] is False
    assert summary["failures"] == [
        {"file": "sell_prices.csv", "check": "prices_positive"}
    ]
