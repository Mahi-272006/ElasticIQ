"""Loading utilities for the raw M5 CSV files.

Every function accepts an optional ``data_dir`` so that tests can point the
loader at small fixture directories. When omitted, the configured raw data
directory (``data/raw`` by default) is used.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import get_settings

# The files expected by ElasticIQ.
EXPECTED_FILES = (
    "calendar.csv",
    "sales_train_validation.csv",
    "sell_prices.csv",
    "sales_train_evaluation.csv",
    "sample_submission.csv",
)

# Backwards-compatible module-level path (resolved from configuration).
RAW_DATA_DIR: Path = get_settings().raw_data_dir


def _resolve_dir(data_dir: Path | None) -> Path:
    return Path(data_dir) if data_dir is not None else RAW_DATA_DIR


def get_csv_path(file_name: str, data_dir: Path | None = None) -> Path:
    """Return the path of an expected CSV file."""

    if file_name not in EXPECTED_FILES:
        raise ValueError(f"Unexpected dataset filename: {file_name}")

    file_path = _resolve_dir(data_dir) / file_name

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Check that the CSV exists inside data/raw/."
        )

    return file_path


def list_raw_files(data_dir: Path | None = None) -> list[dict]:
    """List expected files and report their availability and size."""

    directory = _resolve_dir(data_dir)
    results = []

    for file_name in EXPECTED_FILES:
        file_path = directory / file_name
        exists = file_path.is_file()

        results.append(
            {
                "filename": file_name,
                "exists": exists,
                "size_mb": (
                    round(file_path.stat().st_size / (1024**2), 2) if exists else None
                ),
            }
        )

    return results


def load_csv(
    file_name: str,
    nrows: int | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load an expected CSV file into a Pandas DataFrame.

    Use nrows to load only a sample when exploring large files.
    """

    file_path = get_csv_path(file_name, data_dir)

    try:
        return pd.read_csv(file_path, nrows=nrows)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The CSV file is empty: {file_name}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse CSV file: {file_name}") from exc


def inspect_csv(file_name: str, data_dir: Path | None = None) -> dict:
    """Inspect a small sample without loading the entire CSV."""

    file_path = get_csv_path(file_name, data_dir)
    sample = load_csv(file_name, nrows=5, data_dir=data_dir)

    return {
        "filename": file_name,
        "size_mb": round(file_path.stat().st_size / (1024**2), 2),
        "columns": sample.columns.tolist(),
        "sample_rows": len(sample),
        "sample": sample,
    }


def day_columns(columns: list[str] | pd.Index) -> list[str]:
    """Return the sorted ``d_N`` sales columns from a DataFrame's columns."""

    return sorted(
        (col for col in columns if col.startswith("d_") and col[2:].isdigit()),
        key=lambda col: int(col[2:]),
    )
