"""Final-forecast post-processing: recent level x model shape x series scale.

Why this exists
---------------
Diagnosis (``artifacts/reports/diagnosis*.json``) showed the dominant
*fixable* error of the gradient-boosting model is per-series **level**
miscalibration: it over-predicts sparse series (0.60 units on days that are
actually zero) and under-predicts dense ones.  Experiments on the dev window
(``PROGRESS.md``) showed three findings:

1. the model's day-to-day *pattern* (shape) is better than a plain weekday
   profile, but its per-series *level* is not;
2. replacing the level with the mean of the last few observed days and then
   refitting one scale factor per series on a held-out window recovers
   ~9-12 WAPE points, leak-free;
3. **how much** of that scale refit survives into production depends on
   transfer: scales fitted and applied against the *same* fixed level window
   look great in-sample but fail when production applies them against a
   *fresh* level (the official-window failure documented in ``PROGRESS.md``).

The final forecast is therefore::

    final[id, day] = scale[id] * level[id] * shape[id, day]

    shape  = raw prediction normalised to mean 1 within each series
    level  = mean of the last ``level_window`` observed days of history
    scale  = per-series factor fitted on a held-out window (leak-free),
             shrunk toward the global scale by ``shrink``

Defaults (``level_window=14``, ``shrink=300``) were selected by the
cross-block harness (``PROGRESS.md``, cross-block validation): three
out-of-sample fit->score block pairs plus a fresh-level dev pair, each pair
reproducing production exactly (fresh level computed from each block's own
history).  Across all pairs the shrink curve has an interior minimum near
300 at every level window, and level windows of 14/28 days dominate 7 days
(less sampling noise on intermittent series).  ``shrink=0`` - every series
keeping its own fitted scale - was the worst candidate in every block.

Fitting (``fit_postprocess``) only ever sees days before the evaluation
window, and applying (``apply_postprocess``) only uses observed history plus
the model's own predictions - never future actuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.calibration import fit_per_series_scales

DEFAULT_LEVEL_WINDOW = 14   # two full weekday cycles; beats 7d on cross-block
DEFAULT_SHRINK = 300.0      # weight of the global scale vs a series' own fit
DEFAULT_CLIP = (0.05, 4.0)  # bounds on scale x prior, safety against outliers


def _levelled_values(
    raw_predictions: pd.DataFrame,
    history: pd.DataFrame,
    level_window: int,
) -> pd.Series:
    """``level[id] * shape[id, day]`` aligned to ``raw_predictions`` rows.

    Series whose raw predictions are all zero keep their raw values (a flat
    zero forecast must not be inflated into a positive level), and series
    missing from ``history`` fall back to the model's own mean level.
    """

    if level_window < 1:
        raise ValueError(f"level_window must be >= 1, got {level_window}")

    raw = raw_predictions["predicted_units"].to_numpy(dtype=np.float64)
    ids = raw_predictions["id"]

    # Group positionally so a caller's odd/duplicate index cannot misalign.
    means = pd.Series(raw).groupby(ids.to_numpy()).mean()
    mean = ids.map(means).to_numpy(dtype=np.float64)
    keep = mean > 0

    shape = np.divide(raw, mean, out=np.zeros_like(raw), where=keep)

    last_day = int(history["day_idx"].max())
    recent = history[history["day_idx"] > last_day - level_window]
    level_by_id = recent.groupby("id")["units"].mean()
    level = ids.map(level_by_id).to_numpy(dtype=np.float64)

    # Fallback for ids absent from history: keep the model's own level.
    level = np.where(np.isfinite(level), level, np.where(keep, mean, 0.0))

    values = level * shape
    return pd.Series(np.where(keep, values, raw), index=raw_predictions.index)


def levelled_forecast(
    raw_predictions: pd.DataFrame,
    history: pd.DataFrame,
    level_window: int = DEFAULT_LEVEL_WINDOW,
) -> pd.DataFrame:
    """Copy of ``raw_predictions`` with the level replaced by recent history."""

    out = raw_predictions.copy()
    out["predicted_units"] = _levelled_values(raw_predictions, history, level_window)
    return out


def fit_postprocess(
    raw_predictions: pd.DataFrame,
    history: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    level_window: int = DEFAULT_LEVEL_WINDOW,
    shrink: float = DEFAULT_SHRINK,
    clip: tuple[float, float] = DEFAULT_CLIP,
    fitted_on: dict | None = None,
) -> dict:
    """Fit per-series scale factors on a held-out window.

    Parameters
    ----------
    raw_predictions:
        Raw (unprocessed) recursive forecasts with columns ``id``,
        ``day_idx``, ``predicted_units`` for the fitting window.
    history:
        Panel rows strictly *before* the fitting window (supplies the
        recent-level prior for those days).
    actuals:
        Actuals of the fitting window (``id``, ``day_idx``, ``actual``).
    fitted_on:
        Free-form provenance record (dates, command, ...) stored in the
        bundle so the artifact is auditable.
    """

    frame = raw_predictions[["id", "day_idx", "predicted_units"]].merge(
        actuals, on=["id", "day_idx"], how="inner", validate="one_to_one"
    )
    if frame.empty:
        raise ValueError("No overlapping rows between predictions and actuals")

    frame = frame.copy()
    frame["predicted_units"] = _levelled_values(frame, history, level_window)

    fitted = fit_per_series_scales(frame, shrink=shrink, clip=clip)
    return {
        "level_window": int(level_window),
        "shrink": float(shrink),
        "clip": [float(clip[0]), float(clip[1])],
        "per_series": fitted["per_series"],
        "global_scale": float(fitted["global_scale"]),
        "fitted_rows": int(fitted["rows_used"]),
        "fitted_series": int(fitted["series_count"]),
        "fitted_on": fitted_on or {},
    }


def apply_postprocess(
    raw_predictions: pd.DataFrame,
    history: pd.DataFrame,
    postprocess: dict,
) -> pd.DataFrame:
    """Turn raw model outputs into final forecasts (same schema)."""

    levelled = _levelled_values(
        raw_predictions, history, int(postprocess.get("level_window",
                                                       DEFAULT_LEVEL_WINDOW))
    )
    per_series = postprocess.get("per_series", {})
    default = float(postprocess.get("global_scale", 1.0))
    factors = raw_predictions["id"].map(
        lambda value: float(per_series.get(str(value), default))
    ).to_numpy(dtype=np.float64)

    out = raw_predictions.copy()
    out["predicted_units"] = levelled.to_numpy(dtype=np.float64) * factors
    return out
