"""Forecast error tracking and artifact staleness.

Honesty notes
-------------
The M5 dataset is a *frozen historical* dataset: wall-clock "data staleness"
against today's date is expected and reported as informational, not as an
alert. What matters operationally is (a) whether the trained artifact is old
relative to the data it was trained on, and (b) how predictions scored when
actuals became available.

``score_predictions`` is the reusable scoring path: it accepts any frame with
actual and predicted columns and computes per-horizon WAPE/MAE plus signed
bias (mean error). ``evaluation_report`` replays the stored offline
validation results so the API can serve the already-computed honest numbers
without retraining.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, wape


def score_predictions(frame: pd.DataFrame) -> dict:
    """Score actual vs predicted demand with per-horizon breakdowns.

    ``frame`` needs ``actual_units``, ``predicted_units`` and optionally
    ``horizon`` (steps-ahead) and ``cat_id`` columns.
    """

    required = {"actual_units", "predicted_units"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Scoring frame missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Scoring frame is empty")

    actual = frame["actual_units"].to_numpy(dtype=float)
    predicted = frame["predicted_units"].to_numpy(dtype=float)

    result: dict = {
        "rows": int(len(frame)),
        "overall": {
            "wape_percent": round(wape(actual, predicted), 4),
            "mae_units": round(mae(actual, predicted), 4),
            "mean_bias_units": round(float(np.mean(predicted - actual)), 4),
        },
    }

    if "horizon" in frame.columns:
        by_horizon: dict[str, dict] = {}
        for horizon, group in sorted(frame.groupby("horizon")):
            g_actual = group["actual_units"].to_numpy(dtype=float)
            g_pred = group["predicted_units"].to_numpy(dtype=float)
            by_horizon[str(int(horizon))] = {
                "wape_percent": round(wape(g_actual, g_pred), 4),
                "mae_units": round(mae(g_actual, g_pred), 4),
                "mean_bias_units": round(float(np.mean(g_pred - g_actual)), 4),
                "rows": int(len(group)),
            }
        result["by_horizon"] = by_horizon

    if "cat_id" in frame.columns:
        by_category: dict[str, dict] = {}
        for category, group in sorted(frame.groupby("cat_id")):
            g_actual = group["actual_units"].to_numpy(dtype=float)
            g_pred = group["predicted_units"].to_numpy(dtype=float)
            by_category[str(category)] = {
                "wape_percent": round(wape(g_actual, g_pred), 4),
                "mae_units": round(mae(g_actual, g_pred), 4),
                "rows": int(len(group)),
            }
        result["by_category"] = by_category

    result["note"] = (
        "WAPE is scale-free percentage error; mean bias > 0 means predictions "
        "run high. Metrics describe this sample only."
    )
    return result


def staleness_report(
    models_dir: Path,
    last_data_date: str | datetime,
    model_version: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Report artifact and data staleness with the historical-data caveat.

    ``last_data_date`` is the last date present in the dataset (ISO string or
    datetime). ``now`` defaults to the current UTC time (injectable for tests).
    """

    now = now or datetime.now(tz=UTC)
    if isinstance(last_data_date, str):
        last_data = datetime.fromisoformat(last_data_date)
        if last_data.tzinfo is None:
            last_data = last_data.replace(tzinfo=UTC)
    else:
        last_data = last_data_date
    if last_data.tzinfo is None:
        last_data = last_data.replace(tzinfo=UTC)

    data_age_days = (now - last_data).days

    artifact_info: dict = {"available": False}
    versions_dir = Path(models_dir)
    if model_version is not None:
        bundle_path = versions_dir / model_version / "bundle.joblib"
    else:
        candidates = (
            sorted(
                (p for p in versions_dir.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
            )
            if versions_dir.is_dir()
            else []
        )
        bundle_path = candidates[-1] / "bundle.joblib" if candidates else None

    model_age_days = None
    if bundle_path is not None and bundle_path.is_file():
        trained_at = datetime.fromtimestamp(bundle_path.stat().st_mtime, tz=UTC)
        model_age_days = (now - trained_at).days
        artifact_info = {
            "available": True,
            "version": bundle_path.parent.name,
            "trained_at": trained_at.isoformat(),
            "model_age_days": model_age_days,
        }

    return {
        "dataset_last_date": last_data.isoformat(),
        "dataset_age_days_vs_wall_clock": data_age_days,
        "dataset_staleness_note": (
            "The M5 dataset ends in 2016 by design; a large wall-clock age is "
            "expected for this historical dataset and is NOT an alert condition."
        ),
        "artifact": artifact_info,
        "artifact_age_days": model_age_days,
    }


def evaluation_report(models_dir: Path, version: str) -> dict:
    """Load the stored offline validation evaluation for a model version."""

    path = Path(models_dir) / version / "evaluation.json"
    if not path.is_file():
        raise FileNotFoundError(f"No evaluation.json for version {version}: {path}")

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    payload["replay_note"] = (
        "Offline validation replay on held-out chronological days "
        "(d_1886-d_1913 of the M5 training range); not live production "
        "monitoring, which requires post-deployment actuals."
    )
    return payload
