"""Scoring helpers for the sampling-bias experiment.

All WAPE/MAE numbers come from the project's own definitions
(``src/evaluation/metrics.py``) through the project's own report function
(``src.forecasting.train.evaluate_forecasts``).  Nothing here redefines a
metric; the additions are:

* an **integrity block** that reports, instead of silently dropping, every
  series/row a model failed to predict;
* a **per-store breakdown** (the shipped report only breaks down by category)
  computed with the same ``wape``/``mae`` functions;
* a **delta block** comparing two cells scored on the *same* cohort.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, wape
from src.forecasting.train import evaluate_forecasts

WAPE_DEFINITION = (
    "src/evaluation/metrics.py wape() = sum(|actual - predicted|) / "
    "sum(|actual|) * 100 (0.0 when both sums are 0, inf otherwise)"
)


def split_predictions_with_actuals(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a saved predictions file into (predictions, actuals)."""

    if "actual" not in frame.columns:
        raise ValueError("Saved predictions file has no 'actual' column")
    actuals = frame[["id", "day_idx", "actual"]].copy()
    predictions = frame.drop(columns=["actual"]).copy()
    return predictions, actuals


def _round_metrics(actual, predicted) -> dict:
    return {
        "wape_percent": round(wape(actual, predicted), 4),
        "mae_units": round(mae(actual, predicted), 4),
    }


def score_cell(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    metadata: pd.DataFrame,
    cohort_ids: Sequence[str] | None = None,
) -> dict:
    """Score one (model, cohort) cell.

    Parameters
    ----------
    predictions:
        Columns ``id, day_idx, predicted_units`` (+ optional d/date).
    actuals:
        Columns ``id, day_idx, actual`` for the cohort's evaluation window.
    metadata:
        One row per series indexed by ``id`` with at least ``cat_id``,
        ``dept_id``, ``store_id``.
    cohort_ids:
        The full cohort.  Everything a model should have predicted but did
        not is reported in ``integrity`` rather than dropped silently.
    """

    if cohort_ids is None:
        raise ValueError("cohort_ids is required for the integrity report")
    cohort = set(cohort_ids)

    # Count rows the cohort definition will exclude BEFORE restricting, so
    # nothing is dropped silently - the integrity block reports them.
    extra_series = sorted(set(predictions["id"].unique()) - cohort)
    extra_actual_series = sorted(set(actuals["id"].unique()) - cohort)

    # Restrict both frames to the cohort so the comparison is fixed-series.
    actuals = actuals[actuals["id"].isin(cohort)]
    predictions = predictions[predictions["id"].isin(cohort)]

    days = sorted(int(day) for day in actuals["day_idx"].unique())
    predicted_ids = set(predictions["id"].unique())
    missing_series = sorted(cohort - predicted_ids)

    finite = np.isfinite(predictions["predicted_units"].to_numpy(dtype=np.float64))
    nonfinite = int((~finite).sum())

    matched = predictions.merge(
        actuals, on=["id", "day_idx"], how="inner", validate="one_to_one"
    )

    integrity = {
        "cohort_series": len(cohort),
        "cohort_series_with_predictions": len(cohort & predicted_ids),
        "series_without_predictions": len(missing_series),
        "series_without_predictions_preview": missing_series[:10],
        "predictions_for_series_outside_cohort": len(extra_series),
        "predictions_for_series_outside_cohort_preview": extra_series[:10],
        "actuals_for_series_outside_cohort": len(extra_actual_series),
        "evaluation_days": (
            f"d_{days[0]}..d_{days[-1]}" if days else None
        ),
        "evaluation_day_count": len(days),
        "actual_rows": int(len(actuals)),
        "expected_rows": int(len(cohort) * len(days)),
        "prediction_rows": int(len(predictions)),
        "matched_rows": int(len(matched)),
        "actual_rows_without_prediction": int(len(actuals) - len(matched)),
        "predictions_without_actual": int(len(predictions) - len(matched)),
        "nonfinite_predictions": nonfinite,
        "total_actual_units": float(actuals["actual"].sum()),
        "complete": bool(
            len(matched) == len(actuals) == len(cohort) * len(days)
            and not missing_series
            and not extra_series
            and nonfinite == 0
        ),
    }

    # Official project report (overall / by horizon / by category).
    official = evaluate_forecasts(predictions, actuals, metadata)

    merged = matched.merge(metadata, on="id", how="left", validate="many_to_one")

    by_store: dict[str, dict] = {}
    for store, group in merged.groupby("store_id", sort=True):
        by_store[str(store)] = {
            **_round_metrics(group["actual"], group["predicted_units"]),
            "rows": int(len(group)),
            "series": int(group["id"].nunique()),
            "total_actual_units": float(group["actual"].sum()),
            "denominator_nonzero": bool(group["actual"].sum() > 0),
        }

    by_category: dict[str, dict] = {}
    for category, group in merged.groupby("cat_id", sort=True):
        by_category[str(category)] = {
            **_round_metrics(group["actual"], group["predicted_units"]),
            "rows": int(len(group)),
            "series": int(group["id"].nunique()),
            "total_actual_units": float(group["actual"].sum()),
            "denominator_nonzero": bool(group["actual"].sum() > 0),
        }

    overall = official["overall"]
    return {
        "wape_percent": overall["wape_percent"],
        "mae_units": overall["mae_units"],
        "rows": overall["rows"],
        "series": int(predictions["id"].nunique()),
        "total_actual_units": integrity["total_actual_units"],
        "official": official,
        "breakdowns": {"by_store": by_store, "by_category": by_category},
        "integrity": integrity,
    }


def compare_cells(arm_a: dict, arm_b: dict) -> dict:
    """Delta between two cells scored on the same cohort.

    Negative ``delta_pp`` means Arm B (stratified) has lower WAPE than Arm A.
    Per-store/per-category winners are only reported where both denominators
    are non-zero.
    """

    wape_a = float(arm_a["wape_percent"])
    wape_b = float(arm_b["wape_percent"])
    delta_pp = round(wape_b - wape_a, 4)
    relative = round((wape_b - wape_a) / wape_a * 100.0, 4) if wape_a else None

    if wape_b < wape_a:
        winner = "arm_b"
    elif wape_a < wape_b:
        winner = "arm_a"
    else:
        winner = "tie"

    def _winners(section: str) -> dict:
        left = arm_a["breakdowns"][section]
        right = arm_b["breakdowns"][section]
        table: dict[str, dict] = {}
        for key in sorted(set(left) | set(right)):
            entry: dict = {}
            if key in left and key in right:
                entry["arm_a_wape"] = left[key]["wape_percent"]
                entry["arm_b_wape"] = right[key]["wape_percent"]
                entry["delta_pp"] = round(
                    right[key]["wape_percent"] - left[key]["wape_percent"], 4
                )
                denominators_ok = left[key]["denominator_nonzero"] and right[
                    key
                ]["denominator_nonzero"]
                entry["denominator_nonzero"] = bool(denominators_ok)
                if denominators_ok:
                    if entry["delta_pp"] < 0:
                        entry["winner"] = "arm_b"
                    elif entry["delta_pp"] > 0:
                        entry["winner"] = "arm_a"
                    else:
                        entry["winner"] = "tie"
                else:
                    entry["winner"] = None
            table[key] = entry
        return table

    per_store = _winners("by_store")
    per_category = _winners("by_category")

    def _sweep(table: dict) -> dict:
        decided = [e["winner"] for e in table.values() if e.get("winner")]
        return {
            "arm_b_better_in": sum(1 for w in decided if w == "arm_b"),
            "arm_a_better_in": sum(1 for w in decided if w == "arm_a"),
            "ties": sum(1 for w in decided if w == "tie"),
            "undecided_zero_denominator": sum(
                1 for e in table.values() if e.get("winner") is None
            ),
        }

    return {
        "arm_a_wape_percent": wape_a,
        "arm_b_wape_percent": wape_b,
        "delta_pp": delta_pp,
        "relative_change_percent": relative,
        "arm_a_mae_units": arm_a["mae_units"],
        "arm_b_mae_units": arm_b["mae_units"],
        "mae_delta_units": round(arm_b["mae_units"] - arm_a["mae_units"], 4),
        "winner": winner,
        "rows": arm_a["rows"],
        "series": arm_a["series"],
        "total_actual_units": arm_a["total_actual_units"],
        "cohorts_identical": bool(
            arm_a["series"] == arm_b["series"] and arm_a["rows"] == arm_b["rows"]
        ),
        "per_store": per_store,
        "per_category": per_category,
        "per_store_sweep": _sweep(per_store),
        "per_category_sweep": _sweep(per_category),
    }
