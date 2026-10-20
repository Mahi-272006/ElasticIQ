import json
from pathlib import Path

import numpy as np

from src.data.loader import day_columns, load_csv
from src.evaluation.metrics import mae, wape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"

# Forecast setup
VALIDATION_DAYS = 28
LOOKBACK_DAYS = 28
NUMBER_OF_PRODUCTS = 500


def main() -> None:
    print("Loading sales data...")
    sales = load_csv("sales_train_validation.csv")

    columns = day_columns(sales.columns)

    if len(columns) < LOOKBACK_DAYS + VALIDATION_DAYS:
        raise ValueError("Not enough daily observations for this forecast setup.")

    # Use a deterministic sample of products for an initial experiment.
    sample = sales.head(NUMBER_OF_PRODUCTS)
    print(f"Selected {len(sample):,} product-store series.")

    # Hold out the final 28 days. Do not use them to make predictions.
    training_columns = columns[:-VALIDATION_DAYS]
    validation_columns = columns[-VALIDATION_DAYS:]

    # Baseline: predict each product's mean over its last 28 training days.
    recent_training_columns = training_columns[-LOOKBACK_DAYS:]

    actual = sample[validation_columns].to_numpy(dtype=np.float32)
    recent = sample[recent_training_columns].to_numpy(dtype=np.float32)

    predictions = np.repeat(
        recent.mean(axis=1, keepdims=True),
        VALIDATION_DAYS,
        axis=1,
    )

    wape_percent = wape(actual, predictions)
    mae_units = mae(actual, predictions)

    # Compare with a naive forecast that repeats the last observed day.
    last_day = sample[training_columns[-1]].to_numpy(dtype=np.float32)
    naive_predictions = np.repeat(
        last_day[:, None],
        VALIDATION_DAYS,
        axis=1,
    )
    naive_wape_percent = wape(actual, naive_predictions)

    results = {
        "experiment": "28-day mean baseline",
        "series_count": int(len(sample)),
        "training_days": int(len(training_columns)),
        "validation_days": int(len(validation_columns)),
        "lookback_days": LOOKBACK_DAYS,
        "baseline_wape_percent": round(wape_percent, 4),
        "baseline_mae_units": round(mae_units, 4),
        "last_day_naive_wape_percent": round(naive_wape_percent, 4),
        "note": (
            "Initial benchmark on the first 500 product-store series. "
            "Use the same series and validation dates for future comparisons."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "baseline_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print("ELASTICIQ - FORECASTING BASELINE")
    print("=" * 50)
    print(f"Series evaluated: {results['series_count']:,}")
    print(f"Training days: {results['training_days']:,}")
    print(f"Validation days: {results['validation_days']}")
    print(f"28-day mean WAPE: {wape_percent:.2f}%")
    print(f"28-day mean MAE: {mae_units:.2f} units")
    print(f"Last-day naive WAPE: {naive_wape_percent:.2f}%")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
