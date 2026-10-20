"""Unit tests for the raw-data loader (no full dataset required)."""

from pathlib import Path

import pytest

from src.data.loader import (
    EXPECTED_FILES,
    day_columns,
    get_csv_path,
    inspect_csv,
    list_raw_files,
    load_csv,
)


def test_expected_files_cover_m5_dataset():
    assert set(EXPECTED_FILES) == {
        "calendar.csv",
        "sales_train_validation.csv",
        "sell_prices.csv",
        "sales_train_evaluation.csv",
        "sample_submission.csv",
    }


def test_list_raw_files_reports_availability(raw_dir: Path):
    listing = {item["filename"]: item for item in list_raw_files(data_dir=raw_dir)}

    for name in EXPECTED_FILES:
        assert listing[name]["exists"] is True
        assert listing[name]["size_mb"] is not None


def test_list_raw_files_reports_missing(raw_dir: Path):
    (raw_dir / "calendar.csv").unlink()

    listing = {item["filename"]: item for item in list_raw_files(data_dir=raw_dir)}
    assert listing["calendar.csv"]["exists"] is False
    assert listing["calendar.csv"]["size_mb"] is None


def test_get_csv_path_rejects_unknown_file(raw_dir: Path):
    with pytest.raises(ValueError, match="Unexpected dataset filename"):
        get_csv_path("not_a_dataset.csv", data_dir=raw_dir)


def test_get_csv_path_reports_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        get_csv_path("calendar.csv", data_dir=tmp_path)


def test_load_csv_respects_nrows(raw_dir: Path):
    frame = load_csv("calendar.csv", nrows=3, data_dir=raw_dir)
    assert len(frame) == 3


def test_load_csv_rejects_corrupt_file(tmp_path: Path):
    bad = tmp_path / "calendar.csv"
    bad.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_csv("calendar.csv", data_dir=tmp_path)


def test_inspect_csv_returns_sample(raw_dir: Path):
    info = inspect_csv("calendar.csv", data_dir=raw_dir)

    assert info["sample_rows"] == 5
    assert "wm_yr_wk" in info["columns"]


def test_day_columns_sorts_numerically():
    columns = ["id", "d_10", "d_2", "d_1", "item_id", "d_x"]
    assert day_columns(columns) == ["d_1", "d_2", "d_10"]
