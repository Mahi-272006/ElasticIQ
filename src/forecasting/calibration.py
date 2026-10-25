"""Post-hoc prediction calibration for WAPE.

Diagnosis (artifacts/reports/diagnosis2.json) showed the model's largest
*fixable* error is per-series level miscalibration: rescaling each series'
own predictions optimally would take WAPE from 88.83% to 72.54% (a LEAKY
probe — it used validation actuals, so it is an upper bound on what is
achievable, not a score).

This module fits the same kind of scaling **leak-free**:

1. ``fit``   — estimate scale factors on a *dev* window (d_1858..d_1885),
               which is cut from the training side of the data.
2. ``apply`` — apply those fixed factors to the model's predictions on the
               official window (d_1886..d_1913).  The official actuals are
               never consulted.

Why a weighted median: for a fixed scale s,

    sum |actual - s * predicted| = sum w_i * |ratio_i - s|  with
    w_i = predicted_i and ratio_i = actual_i / predicted_i,

which is minimised exactly when s is the weighted median of the ratios.
That is the *MAE-optimal* scale, and pooled WAPE = MAE / mean(actual), so
it is also the WAPE-optimal scale.

Usage::

    python -m src.forecasting.calibration --fit <dev_predictions.parquet>
    python -m src.forecasting.calibration --apply <predictions.parquet> \
        --calibration <calibration.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import wape

DEFAULT_SCALE = 1.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median (any minimiser of sum w_i |v_i - m|)."""

    if values.size == 0:
        return DEFAULT_SCALE
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    if cumulative[-1] <= 0:
        return DEFAULT_SCALE
    cutoff = cumulative[-1] / 2.0
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(v[min(idx, v.size - 1)])


def fit_global_scale(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Single scale minimising pooled absolute error."""

    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    positive = p > 1e-9
    if not positive.any():
        return DEFAULT_SCALE
    return _weighted_median(a[positive] / p[positive], p[positive])


def fit_per_series_scales(
    predictions: pd.DataFrame,
    *,
    shrink: float = 30.0,
    clip: tuple[float, float] = (0.05, 4.0),
    min_series_rows: int = 7,
) -> dict:
    """Per-series scale shrunk toward the global scale.

    Parameters
    ----------
    predictions:
        DataFrame with columns ``id``, ``actual``, ``predicted_units``.
    shrink:
        Pseudo-volume in predicted units at which a series' own estimate
        counts half-way between its own value and the global scale.  Sparse
        series (little evidence) stay near the global scale; dense series
        get their own level.  Selected on the dev window only.
    clip:
        Safety bounds so a single odd series cannot explode.
    min_series_rows:
        Series with fewer rows fall back to the global scale.
    """

    frame = predictions[["id", "actual", "predicted_units"]].copy()
    frame = frame[frame["predicted_units"].notna() & frame["actual"].notna()]

    global_scale = fit_global_scale(
        frame["actual"].to_numpy(), frame["predicted_units"].to_numpy()
    )

    scales: dict[str, float] = {}
    for series_id, group in frame.groupby("id", sort=False):
        if len(group) < min_series_rows:
            scales[str(series_id)] = global_scale
            continue
        p = group["predicted_units"].to_numpy(dtype=np.float64)
        a = group["actual"].to_numpy(dtype=np.float64)
        positive = p > 1e-9
        if not positive.any():
            scales[str(series_id)] = global_scale
            continue
        own = _weighted_median(
            np.clip(a[positive] / p[positive], clip[0], clip[1]), p[positive]
        )
        volume = float(p.sum())
        w = volume / (volume + shrink) if volume > 0 else 0.0
        scales[str(series_id)] = float(w * own + (1.0 - w) * global_scale)

    return {
        "global_scale": global_scale,
        "per_series": scales,
        "shrink": shrink,
        "clip": list(clip),
        "rows_used": int(len(frame)),
        "series_count": int(frame["id"].nunique()),
    }


def apply_scales(
    predictions: pd.DataFrame, calibration: dict
) -> pd.DataFrame:
    """Return a copy with ``predicted_units`` rescaled by the fitted factors."""

    out = predictions.copy()
    per_series = calibration.get("per_series", {})
    default = float(calibration.get("global_scale", DEFAULT_SCALE))
    factors = out["id"].map(lambda value: per_series.get(str(value), default))
    out["predicted_units_raw"] = predictions["predicted_units"]
    out["predicted_units"] = (
        out["predicted_units"].to_numpy(dtype=np.float64)
        * np.asarray(factors, dtype=np.float64)
    )
    return out


def evaluate(predictions: pd.DataFrame) -> dict:
    """WAPE/MAE of a predictions frame with an ``actual`` column."""

    return {
        "wape_percent": round(
            wape(predictions["actual"], predictions["predicted_units"]), 4
        ),
        "rows": int(len(predictions)),
        "bias": round(
            float(
                predictions["predicted_units"].sum()
                / max(predictions["actual"].sum(), 1e-9)
                - 1.0
            ),
            4,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit or apply WAPE calibration")
    parser.add_argument("--fit", dest="fit_path", help="dev predictions parquet")
    parser.add_argument("--apply", dest="apply_path", help="predictions to rescale")
    parser.add_argument("--calibration", default=None,
                        help="calibration.json (output of --fit, input to --apply)")
    parser.add_argument("--shrink", type=float, default=30.0)
    parser.add_argument("--out", default=None, help="where to write calibration json")
    parser.add_argument("--predictions-out", default=None,
                        help="where to write the rescaled predictions parquet")
    args = parser.parse_args()

    if args.fit_path:
        dev = pd.read_parquet(args.fit_path)
        before = evaluate(dev)
        calibration = fit_per_series_scales(dev, shrink=args.shrink)
        calibration["fitted_on"] = str(args.fit_path)
        after = evaluate(apply_scales(dev, calibration))
        out = Path(args.out or Path(args.fit_path).parent / "calibration.json")
        out.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        print(f"FIT on {args.fit_path}")
        print(f"  shrink={args.shrink}  global_scale="
              f"{calibration['global_scale']:.4f}  "
              f"series={calibration['series_count']}")
        print(f"  WAPE before: {before['wape_percent']}%  bias {before['bias']}")
        print(f"  WAPE after : {after['wape_percent']}%  bias {after['bias']}")
        print(f"  saved -> {out}")
        return

    if args.apply_path and args.calibration:
        preds = pd.read_parquet(args.apply_path)
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        before = evaluate(preds)
        rescaled = apply_scales(preds, calibration)
        after = evaluate(rescaled)
        out = Path(
            args.predictions_out
            or Path(args.apply_path).with_name("predictions_calibrated.parquet")
        )
        rescaled.to_parquet(out, index=False)
        print(f"APPLY {args.calibration} -> {args.apply_path}")
        print(f"  WAPE before: {before['wape_percent']}%  bias {before['bias']}")
        print(f"  WAPE after : {after['wape_percent']}%  bias {after['bias']}")
        print(f"  saved -> {out}")
        return

    parser.error("provide --fit (to fit) or --apply with --calibration (to apply)")


if __name__ == "__main__":
    main()
