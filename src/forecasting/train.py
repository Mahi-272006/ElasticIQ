"""Train and evaluate the demand forecasting model.

Design decisions (recorded in the model metadata):

- Chronological split: the final 28 days of the panel are the validation
  window; all earlier days are training data.
- Training rows use only past information (lags and rolling windows are
  shifted), so no future sales leak into features.
- Validation uses *recursive* inference: each predicted day is fed back
  into the feature history, exactly like deployment. Validation actuals are
  never used as model inputs.
- Future sell prices during validation are assumed equal to the last
  observed price (the same assumption the API uses at inference time).
- Baselines are recomputed on the same series and dates for fair comparison.

Usage::

    python -m src.forecasting.train [--max-iter 100]

Artifacts are written to ``artifacts/models/<version>/``:
``bundle.joblib`` (model + feature metadata), ``metadata.json``,
``evaluation.json`` and ``validation_predictions.parquet``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import get_settings
from src.data.pipeline import load_processed
from src.evaluation.metrics import mae, wape
from src.features.engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    build_features,
)
from src.forecasting.baselines import (
    last_day_naive,
    mean_baseline,
    weekly_seasonal_naive,
)
from src.forecasting.inference import recursive_forecast

logger = logging.getLogger(__name__)

MODEL_VERSION = "hgb_poisson_v1"
VALIDATION_DAYS = 28
FORECAST_HORIZONS = (7, 14, 28)
INFERENCE_TAIL_DAYS = 60  # history window handed to recursive inference


def chronological_split(
    day_idx: pd.Series, validation_days: int = VALIDATION_DAYS
) -> tuple[int, int]:
    """Return (origin_day, first_validation_day) for a panel."""

    last_day = int(day_idx.max())
    first_validation = last_day - validation_days + 1
    return first_validation - 1, first_validation


def train_model(
    features: pd.DataFrame,
    train_mask: pd.Series,
    max_iter: int,
    random_seed: int,
) -> HistGradientBoostingRegressor:
    """Fit the gradient-boosting model on training rows only."""

    train = features[train_mask]
    X = train[FEATURE_COLUMNS]
    y = train["units"].astype(np.float64)

    categorical_indices = [
        FEATURE_COLUMNS.index(column) for column in CATEGORICAL_FEATURES
    ]

    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=max_iter,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.05,
        n_iter_no_change=8,
        random_state=random_seed,
    )
    model.set_params(categorical_features=categorical_indices)
    model.fit(X, y)

    logger.info(
        "Model fitted: %d iterations, train rows %s",
        model.n_iter_,
        f"{int(train_mask.sum()):,}",
    )
    return model


def matrix_baselines(history: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Baseline forecasts from a (series x history-days) matrix."""

    return {
        "mean_baseline": mean_baseline(history, horizon),
        "last_day_naive": last_day_naive(history, horizon),
        "weekly_seasonal_naive": weekly_seasonal_naive(history, horizon),
    }


def evaluate_forecasts(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    metadata: pd.DataFrame,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
) -> dict:
    """WAPE/MAE overall, by horizon, and by category."""

    merged = predictions.merge(actuals, on=["id", "day_idx"], validate="one_to_one")
    merged = merged.merge(metadata, on="id", how="left", validate="many_to_one")
    origin_day = int(merged["day_idx"].min()) - 1

    def _metrics(frame: pd.DataFrame) -> dict:
        return {
            "wape_percent": round(wape(frame["actual"], frame["predicted_units"]), 4),
            "mae_units": round(mae(frame["actual"], frame["predicted_units"]), 4),
            "rows": int(len(frame)),
        }

    report: dict = {"overall": _metrics(merged), "by_horizon": {}, "by_category": {}}

    for horizon in horizons:
        subset = merged[merged["day_idx"] <= origin_day + horizon]
        report["by_horizon"][str(horizon)] = _metrics(subset)

    for category, group in merged.groupby("cat_id"):
        report["by_category"][str(category)] = _metrics(group)

    return report


def run_training(
    max_iter: int = 100,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict:
    """Train, evaluate, and persist the forecasting model.

    The series sample is decided upstream by the data pipeline (the
    processed ``sample_panel``); this function trains on whatever the
    panel contains and records the selection in metadata.
    """

    settings = get_settings()
    processed_dir = (
        Path(processed_dir) if processed_dir else settings.processed_data_dir
    )
    models_dir = Path(models_dir) if models_dir else settings.models_dir

    started = time.perf_counter()

    logger.info("Loading processed panel...")
    panel = load_processed("sample_panel", processed_dir)
    calendar = load_processed("calendar", processed_dir)

    origin_day, first_validation_day = chronological_split(panel["day_idx"])
    logger.info(
        "Split: train through d_%d, validate d_%d onward (%d days)",
        origin_day,
        first_validation_day,
        VALIDATION_DAYS,
    )

    logger.info("Building features (leakage-safe)...")
    features = build_features(panel)

    train_mask = (features["day_idx"] <= origin_day) & (features["day_idx"] > 28)
    # Only rows with full feature history (no NaN lags) enter training.
    train_mask &= features[FEATURE_COLUMNS].notna().all(axis=1)

    model = train_model(features, train_mask, max_iter, settings.random_seed)

    # ---------------------------------------------- recursive evaluation ---
    logger.info("Recursive validation forecast (%d days)...", VALIDATION_DAYS)
    tail = panel[
        panel["day_idx"].between(origin_day - INFERENCE_TAIL_DAYS + 1, origin_day)
    ]
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_categories": {
            column: list(features[column].cat.categories)
            for column in CATEGORICAL_FEATURES
        },
    }

    predictions = recursive_forecast(bundle, tail, calendar, VALIDATION_DAYS)

    validation_mask = panel["day_idx"] >= first_validation_day
    actuals = panel.loc[validation_mask, ["id", "day_idx", "units"]].rename(
        columns={"units": "actual"}
    )
    metadata = panel.groupby("id", sort=False)[
        ["cat_id", "dept_id", "store_id"]
    ].first()

    ml_report = evaluate_forecasts(predictions, actuals, metadata)

    # ----------------------------------------------------- baseline parity ---
    logger.info("Computing baselines on identical series and dates...")
    history_days = panel["day_idx"] <= origin_day
    history = (
        panel.loc[history_days]
        .pivot(index="id", columns="day_idx", values="units")
        .sort_index()
    )
    actual_matrix = (
        panel.loc[~history_days]
        .pivot(index="id", columns="day_idx", values="units")
        .sort_index()
        .to_numpy(dtype=np.float64)
    )
    history_values = history.to_numpy(dtype=np.float64)

    baseline_reports = {}
    for name, forecast_matrix in matrix_baselines(
        history_values, VALIDATION_DAYS
    ).items():
        baseline_reports[name] = {
            "overall": {
                "wape_percent": round(wape(actual_matrix, forecast_matrix), 4),
                "mae_units": round(mae(actual_matrix, forecast_matrix), 4),
            },
            "by_horizon": {
                str(horizon): {
                    "wape_percent": round(
                        wape(actual_matrix[:, :horizon], forecast_matrix[:, :horizon]),
                        4,
                    ),
                    "mae_units": round(
                        mae(actual_matrix[:, :horizon], forecast_matrix[:, :horizon]), 4
                    ),
                }
                for horizon in FORECAST_HORIZONS
            },
        }

    # ------------------------------------------------------ persist outputs ---
    version_dir = models_dir / MODEL_VERSION
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, version_dir / "bundle.joblib")

    predictions_out = predictions.merge(
        actuals, on=["id", "day_idx"], how="left", validate="one_to_one"
    )
    predictions_out.to_parquet(
        version_dir / "validation_predictions.parquet", index=False
    )

    evaluation = {
        "model": ml_report,
        "baselines": baseline_reports,
        "comparison_note": (
            "All models are evaluated on the same series, the same validation "
            "dates, and with the same recursive 28-day protocol."
        ),
    }
    (version_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )

    metadata_doc = {
        "version": MODEL_VERSION,
        "trained_at_utc": datetime.now(tz=UTC).isoformat(),
        "algorithm": "HistGradientBoostingRegressor",
        "loss": "poisson",
        "hyperparameters": {
            "learning_rate": 0.08,
            "max_iter_requested": max_iter,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "early_stopping": True,
            "random_seed": settings.random_seed,
        },
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "training": {
            "sample_selection": (
                f"first {panel['id'].nunique():,} series in file order "
                "(deterministic head selection from the processed panel)"
            ),
            "series": int(panel["id"].nunique()),
            "train_rows": int(train_mask.sum()),
            "train_days": f"d_1..d_{origin_day}",
            "validation_days": (
                f"d_{first_validation_day}..d_{int(panel['day_idx'].max())}"
            ),
        },
        "features": {
            "numeric": [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES],
            "categorical": CATEGORICAL_FEATURES,
            "leakage_controls": [
                "lags and rolling windows shift(1) before rolling",
                "recursive inference feeds predictions back, never validation actuals",
                "calendar features (weekday, month, SNAP, events) are known in advance",
            ],
        },
        "assumptions": [
            "future sell prices equal the last observed price unless supplied",
            "future SNAP/event calendar known (M5 calendar is fully observed)",
        ],
        "supported_horizons": list(FORECAST_HORIZONS),
        "evaluation_summary": ml_report["by_horizon"],
    }
    (version_dir / "metadata.json").write_text(
        json.dumps(metadata_doc, indent=2), encoding="utf-8"
    )

    elapsed = round(time.perf_counter() - started, 1)
    logger.info("Training complete in %s s; artifacts at %s", elapsed, version_dir)

    return {
        "version": MODEL_VERSION,
        "artifacts_dir": str(version_dir),
        "elapsed_seconds": elapsed,
        "evaluation": evaluation,
        "metadata": metadata_doc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ElasticIQ forecast model")
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    result = run_training(max_iter=args.max_iter)

    print("\n" + "=" * 60)
    print("ELASTICIQ - MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Version: {result['version']}")
    print(f"Artifacts: {result['artifacts_dir']}")
    print(f"Elapsed: {result['elapsed_seconds']} s")

    print("\nValidation WAPE by horizon (recursive 28-day protocol):")
    model_horizons = result["evaluation"]["model"]["by_horizon"]
    for horizon, metrics in model_horizons.items():
        baseline = result["evaluation"]["baselines"]["weekly_seasonal_naive"]
        seasonal_wape = baseline["by_horizon"][horizon]["wape_percent"]
        print(
            f"  h={horizon:>2}: model {metrics['wape_percent']:8.2f}%  "
            f"seasonal-naive {seasonal_wape:8.2f}%"
        )


if __name__ == "__main__":
    main()
