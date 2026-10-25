"""Run one forecasting experiment on a chronological window.

Protocol
--------
Two windows are available:

- ``--eval dev``      train d_1..d_1857, forecast/evaluate d_1858..d_1885.
                      This is the window used to *make decisions* (it is cut
                      from the training side, so the official validation set
                      is not spent on tuning).
- ``--eval official`` train d_1..d_1885, forecast/evaluate d_1886..d_1913.
                      Run this ONCE for the chosen configuration.

Everything else mirrors ``src.forecasting.train`` exactly: same recursive
inference (predictions fed back, never actuals), same 60-day history tail,
same future-price assumption, same WAPE/MAE implementation, baselines
recomputed on identical series and dates.

Outputs (per tag) under ``artifacts/experiments/<tag>/``:

- ``predictions.parquet``  id, day_idx, predicted_units, actual, wday
- ``metrics.json``         overall / by horizon / by category + baselines + config

Usage::

    python -m src.forecasting.experiment --tag e0_base --eval dev
    python -m src.forecasting.experiment --tag e1_l1 --eval dev --loss absolute_error
    python -m src.forecasting.experiment --tag final --eval official \
        --features extended --loss poisson --max-iter 200 --learning-rate 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import get_settings
from src.data.pipeline import load_processed
from src.evaluation.metrics import mae, wape
from src.features.engineering import (
    build_features,
    categorical_columns,
    feature_columns,
)
from src.forecasting.baselines import (
    last_day_naive,
    mean_baseline,
    weekly_seasonal_naive,
)
from src.forecasting.inference import recursive_forecast
from src.forecasting.train import evaluate_forecasts

logger = logging.getLogger(__name__)

# (train_origin, eval_start, eval_end) per evaluation window
WINDOWS = {
    "dev": (1857, 1858, 1885),
    "official": (1885, 1886, 1913),
}
HORIZONS = (7, 14, 28)
INFERENCE_TAIL_DAYS = 60  # must match src.forecasting.train


def fit_model(
    features: pd.DataFrame,
    train_mask: pd.Series,
    columns: list[str],
    cats: list[str],
    *,
    loss: str,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    early_stopping: bool,
    seed: int,
) -> HistGradientBoostingRegressor:
    """Fit HGB with the requested configuration on training rows only."""

    train = features[train_mask]
    X = train[columns]
    y = train["units"].astype(np.float64)

    model = HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        early_stopping=early_stopping,
        validation_fraction=0.05,
        n_iter_no_change=8,
        random_state=seed,
    )
    model.set_params(
        categorical_features=[columns.index(c) for c in cats],
    )
    model.fit(X, y)
    logger.info("Fitted %s model: %d iterations, train rows %s",
                loss, model.n_iter_, f"{int(train_mask.sum()):,}")
    return model


def run_experiment(
    tag: str,
    eval_window: str,
    *,
    loss: str = "poisson",
    learning_rate: float = 0.08,
    max_iter: int = 100,
    max_leaf_nodes: int = 63,
    min_samples_leaf: int = 20,
    early_stopping: bool = True,
    extended: bool = False,
    seed: int | None = None,
    out_root: Path | None = None,
) -> dict:
    """Train, recursively forecast, and evaluate one configuration."""

    settings = get_settings()
    seed = settings.random_seed if seed is None else seed
    train_origin, eval_start, eval_end = WINDOWS[eval_window]

    started = time.perf_counter()
    panel = load_processed("sample_panel", settings.processed_data_dir)
    calendar = load_processed("calendar", settings.processed_data_dir)

    columns = feature_columns(extended=extended)
    cats = categorical_columns(extended=extended)

    # ------------------------------ features + training (leakage-safe) -----
    train_panel = panel[panel["day_idx"] <= train_origin]
    features = build_features(train_panel)
    for column in cats:
        # wday/month are numeric in the shared builder; the extended runs
        # treat them as discrete levels.
        if column in ("wday", "month"):
            features[column] = features[column].astype("category")

    train_mask = (features["day_idx"] <= train_origin) & (features["day_idx"] > 28)
    train_mask &= features[columns].notna().all(axis=1)

    model = fit_model(
        features,
        train_mask,
        columns,
        cats,
        loss=loss,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        early_stopping=early_stopping,
        seed=seed,
    )

    # ------------------------------ recursive forecast ---------------------
    tail = panel[
        panel["day_idx"].between(train_origin - INFERENCE_TAIL_DAYS + 1, train_origin)
    ]
    bundle = {
        "model": model,
        "feature_columns": columns,
        "categorical_categories": {
            column: list(features[column].cat.categories) for column in cats
        },
    }
    horizon = eval_end - eval_start + 1
    predictions = recursive_forecast(bundle, tail, calendar, horizon)

    # ------------------------------ evaluation -----------------------------
    validation_mask = panel["day_idx"].between(eval_start, eval_end)
    actuals = (
        panel.loc[validation_mask, ["id", "day_idx", "units"]]
        .rename(columns={"units": "actual"})
    )
    metadata = panel.groupby("id", sort=False)[
        ["cat_id", "dept_id", "store_id"]
    ].first()

    report = evaluate_forecasts(predictions, actuals, metadata, horizons=HORIZONS)

    # baselines on the identical series/dates
    history_days = panel["day_idx"] <= train_origin
    history = (
        panel.loc[history_days]
        .pivot(index="id", columns="day_idx", values="units")
        .sort_index()
    )
    actual_matrix = (
        panel.loc[validation_mask]
        .pivot(index="id", columns="day_idx", values="units")
        .sort_index()
        .to_numpy(dtype=np.float64)
    )
    history_values = history.to_numpy(dtype=np.float64)
    baseline_reports = {}
    for name, forecast_matrix in {
        "mean_baseline": mean_baseline(history_values, horizon),
        "last_day_naive": last_day_naive(history_values, horizon),
        "weekly_seasonal_naive": weekly_seasonal_naive(history_values, horizon),
    }.items():
        baseline_reports[name] = {
            "overall": {
                "wape_percent": round(wape(actual_matrix, forecast_matrix), 4),
                "mae_units": round(mae(actual_matrix, forecast_matrix), 4),
            },
            "by_horizon": {
                str(h): {
                    "wape_percent": round(
                        wape(actual_matrix[:, :h], forecast_matrix[:, :h]), 4
                    ),
                    "mae_units": round(
                        mae(actual_matrix[:, :h], forecast_matrix[:, :h]), 4
                    ),
                }
                for h in HORIZONS
            },
        }

    # ------------------------------ persist --------------------------------
    out_dir = (out_root or (settings.artifacts_dir / "experiments")) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = predictions.merge(actuals, on=["id", "day_idx"], how="left",
                               validate="one_to_one")
    wdays = merged[["day_idx"]].merge(
        calendar[["day_idx", "wday", "date"]], on="day_idx", how="left"
    )
    merged["wday"] = wdays["wday"].to_numpy()
    merged["date"] = wdays["date"].to_numpy()
    merged.to_parquet(out_dir / "predictions.parquet", index=False)

    config = {
        "tag": tag,
        "eval_window": eval_window,
        "train_days": f"d_1..d_{train_origin}",
        "eval_days": f"d_{eval_start}..d_{eval_end}",
        "loss": loss,
        "learning_rate": learning_rate,
        "max_iter_requested": max_iter,
        "max_iter_fitted": int(model.n_iter_),
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "early_stopping": early_stopping,
        "features": "extended" if extended else "base",
        "feature_count": len(columns),
        "random_seed": seed,
        "series": int(panel["id"].nunique()),
        "train_rows": int(train_mask.sum()),
        "eval_rows": int(len(merged)),
    }
    metrics = {
        "config": config,
        "model": report,
        "baselines": baseline_reports,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # also keep the bundle for later inspection (never overwrites hgb_poisson_v1)
    joblib.dump(bundle, out_dir / "bundle.joblib")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one forecasting experiment")
    parser.add_argument("--tag", required=True, help="experiment name (output folder)")
    parser.add_argument("--eval", dest="eval_window", choices=sorted(WINDOWS),
                        default="dev")
    parser.add_argument("--loss", default="poisson",
                        help="sklearn HGB loss (poisson | absolute_error | ...)")
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--features", choices=["base", "extended"], default="base")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    metrics = run_experiment(
        args.tag,
        args.eval_window,
        loss=args.loss,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        early_stopping=not args.no_early_stopping,
        extended=args.features == "extended",
        seed=args.seed,
    )

    model = metrics["model"]
    print("\n" + "=" * 68)
    print(f"EXPERIMENT {metrics['config']['tag']}  "
          f"[{metrics['config']['eval_window']} window, "
          f"{metrics['config']['features']} features, "
          f"loss={metrics['config']['loss']}]")
    print("=" * 68)
    print(f"  fitted iterations : {metrics['config']['max_iter_fitted']}")
    print(f"  train rows        : {metrics['config']['train_rows']:,}")
    print(f"  eval rows         : {metrics['config']['eval_rows']:,}")
    print(f"  elapsed           : {metrics['elapsed_seconds']} s")
    print("\n  WAPE by horizon (model vs mean baseline):")
    for h, m in model["by_horizon"].items():
        base = metrics["baselines"]["mean_baseline"]["by_horizon"][h]
        print(f"    h={h:>2}: model {m['wape_percent']:8.4f}%   "
              f"mean-baseline {base['wape_percent']:8.4f}%")
    print(f"\n  OVERALL: model {model['overall']['wape_percent']:.4f}%  "
          f"mae {model['overall']['mae_units']}")
    for name, b in metrics["baselines"].items():
        print(f"            {name}: {b['overall']['wape_percent']:.4f}%")
    print("\n  by category:")
    for cat, m in model["by_category"].items():
        print(f"    {cat:<12} {m['wape_percent']:8.4f}%  rows {m['rows']:,}")
    print(f"\n  artifacts -> artifacts/experiments/{metrics['config']['tag']}/")


if __name__ == "__main__":
    main()
