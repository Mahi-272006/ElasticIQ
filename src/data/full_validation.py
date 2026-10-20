import pandas as pd

from src.data.loader import EXPECTED_FILES, RAW_DATA_DIR
from src.data.validation import REQUIRED_COLUMNS

CHUNK_SIZE = 50_000


def validate_full_dataset(file_name: str) -> dict:
    """Scan a CSV in chunks and summarize basic data quality."""

    file_path = RAW_DATA_DIR / file_name

    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    header = pd.read_csv(file_path, nrows=0)
    columns = list(header.columns)

    missing_columns = sorted(REQUIRED_COLUMNS[file_name] - set(columns))

    if missing_columns:
        return {
            "file": file_name,
            "status": "FAIL",
            "reason": f"Missing required columns: {missing_columns}",
        }

    row_count = 0
    missing_by_column = {column: 0 for column in columns}
    duplicate_rows_in_chunks = 0
    invalid_value_count = 0

    # For key uniqueness, keep track of keys seen across all chunks.
    seen_price_keys = set() if file_name == "sell_prices.csv" else None
    duplicate_price_keys = 0

    seen_ids = (
        set()
        if file_name
        in {
            "sales_train_validation.csv",
            "sales_train_evaluation.csv",
            "sample_submission.csv",
        }
        else None
    )
    duplicate_ids = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        row_count += len(chunk)

        for column in columns:
            missing_by_column[column] += int(chunk[column].isna().sum())

        duplicate_rows_in_chunks += int(chunk.duplicated().sum())

        if file_name in {
            "sales_train_validation.csv",
            "sales_train_evaluation.csv",
        }:
            day_columns = [
                column
                for column in chunk.columns
                if column.startswith("d_") and column[2:].isdigit()
            ]

            sales = chunk[day_columns].apply(pd.to_numeric, errors="coerce")

            invalid_value_count += int((sales.isna() | sales.lt(0)).sum().sum())

        elif file_name == "sell_prices.csv":
            prices = pd.to_numeric(chunk["sell_price"], errors="coerce")

            invalid_value_count += int((prices.isna() | prices.le(0)).sum())

            keys = zip(
                chunk["store_id"],
                chunk["item_id"],
                chunk["wm_yr_wk"],
                strict=False,
            )

            for key in keys:
                if key in seen_price_keys:
                    duplicate_price_keys += 1
                else:
                    seen_price_keys.add(key)

        if seen_ids is not None:
            for identifier in chunk["id"]:
                if identifier in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(identifier)

    missing_summary = {
        column: count for column, count in missing_by_column.items() if count > 0
    }

    return {
        "file": file_name,
        "rows": row_count,
        "columns": len(columns),
        "missing_values": missing_summary,
        "duplicate_rows_within_chunks": duplicate_rows_in_chunks,
        "duplicate_price_keys": duplicate_price_keys,
        "duplicate_ids": duplicate_ids,
        "invalid_value_count": invalid_value_count,
        "status": "SCAN COMPLETE",
    }


def run_full_validation() -> list[dict]:
    """Scan every expected CSV file."""

    reports = []

    for file_name in EXPECTED_FILES:
        print(f"Scanning {file_name} ...")
        reports.append(validate_full_dataset(file_name))

    return reports


if __name__ == "__main__":
    for report in run_full_validation():
        print("\n", report)
