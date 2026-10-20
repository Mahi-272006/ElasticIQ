import json
from pathlib import Path

import numpy as np

from src.data.loader import day_columns, load_csv
from src.evaluation.metrics import wape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"

VALIDATION_DAYS = 28
SEASONAL_PERIOD = 7
NUMBER_OF_PRODUCTS = 500


def main() -> None:
    print("Loading sales data...")
    sales = load_csv("sales_train_validation.csv")

    columns = day_columns(sales.columns)

    if len(columns) < VALIDATION_DAYS + SEASONAL_PERIOD:
        raise ValueError("Not enough data for seasonal forecasting.")

    sample = sales.head(NUMBER_OF_PRODUCTS)

    training_columns = columns[:-VALIDATION_DAYS]
    validation_columns = columns[-VALIDATION_DAYS:]

    # Last 7 observed training days, repeated across the 28-day horizon.
    last_week = sample[training_columns[-SEASONAL_PERIOD:]].to_numpy(dtype=np.float32)
    predictions = np.tile(
        last_week,
        (1, VALIDATION_DAYS // SEASONAL_PERIOD),
    )

    actual = sample[validation_columns].to_numpy(dtype=np.float32)
    wape_percent = wape(actual, predictions)

    results = {
        "experiment": "weekly seasonal naive",
        "series_count": int(len(sample)),
        "training_days": int(len(training_columns)),
        "validation_days": int(len(validation_columns)),
        "seasonal_period_days": SEASONAL_PERIOD,
        "wape_percent": round(wape_percent, 4),
        "note": (
            "Same first 500 series and same final 28 validation days "
            "as the initial baseline."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "seasonal_baseline_results.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("ELASTICIQ - WEEKLY SEASONAL BASELINE")
    print("=" * 50)
    print(f"Series evaluated: {len(sample):,}")
    print(f"Validation days: {VALIDATION_DAYS}")
    print(f"Weekly seasonal WAPE: {wape_percent:.2f}%")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
