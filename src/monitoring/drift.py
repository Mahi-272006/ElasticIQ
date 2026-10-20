"""Distribution drift monitoring (Population Stability Index).

What this measures
------------------
PSI compares a *reference* distribution to a *current* distribution of the
same quantity. It answers "has the shape of demand/pricing shifted?", not
"why did it shift" — no causal interpretation is claimed.

Applied here to data the M5 dataset actually contains:

- per-series mean daily units in the most recent ``window_days`` versus the
  preceding ``window_days`` (computed from ``series_history_tail``, which
  stores the last 56 observed days for all 30,490 series)
- per-series first observed price versus last observed price
  (computed from ``price_stats``)

Thresholds follow the common rule of thumb (PSI < 0.10 stable, < 0.25
moderate, above that significant). They are heuristics, not guarantees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_DEFAULT_BINS = 10
PSI_STABLE_BELOW = 0.10
PSI_MODERATE_BELOW = 0.25

_MIN_SAMPLES_PER_BIN = 10
_EPS = 1e-6


def classify_psi(value: float) -> str:
    """Map a PSI value to a coarse stability label (rule of thumb)."""

    if value < PSI_STABLE_BELOW:
        return "stable"
    if value < PSI_MODERATE_BELOW:
        return "moderate"
    return "significant"


def population_stability_index(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    bins: int = PSI_DEFAULT_BINS,
) -> float:
    """Quantile-bin PSI of ``current`` against ``reference``.

    Bin edges come from the reference distribution (deciles by default).
    Zero-mass bins are floored at ``eps`` to keep the log finite, the
    standard practical treatment.
    """

    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) < bins * _MIN_SAMPLES_PER_BIN:
        raise ValueError(
            f"Need at least {bins * _MIN_SAMPLES_PER_BIN} reference samples "
            f"for {bins} bins, got {len(ref)}"
        )
    if len(cur) == 0:
        raise ValueError("Current window has no finite samples")

    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        # Degenerate (near-constant) reference: fall back to min/max split.
        edges = np.array([ref.min() - 1e-9, ref.max() + 1e-9])
    edges = edges.astype(float).copy()
    edges[0] -= _EPS
    edges[-1] += _EPS

    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_share = np.histogram(cur, bins=edges)[0] / len(cur)

    ref_share = np.clip(ref_share, _EPS, None)
    cur_share = np.clip(cur_share, _EPS, None)

    psi = float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
    return psi


def _tail_day_columns(tail: pd.DataFrame) -> list[str]:
    """Sorted d_N day columns of a wide history-tail frame."""

    return sorted(
        (col for col in tail.columns if col.startswith("d_")),
        key=lambda col: int(col[2:]),
    )


def demand_window_drift(tail: pd.DataFrame, window_days: int = 28) -> dict:
    """PSI of per-series mean daily units: recent window vs prior window.

    ``tail`` is the wide ``series_history_tail`` frame (one row per series,
    ``d_N`` day columns ending at the forecast origin).
    """

    day_cols = _tail_day_columns(tail)
    if len(day_cols) < 2 * window_days:
        raise ValueError(
            f"History tail covers {len(day_cols)} days; need at least "
            f"{2 * window_days} for a {window_days}-day drift comparison"
        )

    prior_cols = day_cols[-2 * window_days : -window_days]
    recent_cols = day_cols[-window_days:]

    prior_mean = tail[prior_cols].mean(axis=1).to_numpy(dtype=float)
    recent_mean = tail[recent_cols].mean(axis=1).to_numpy(dtype=float)

    psi = population_stability_index(prior_mean, recent_mean)

    zero_share_prior = float((tail[prior_cols] == 0).to_numpy().mean())
    zero_share_recent = float((tail[recent_cols] == 0).to_numpy().mean())

    return {
        "metric": "per_series_mean_daily_units",
        "window_days": window_days,
        "reference_window": [prior_cols[0], prior_cols[-1]],
        "current_window": [recent_cols[0], recent_cols[-1]],
        "series_compared": int(len(tail)),
        "psi": round(psi, 6),
        "classification": classify_psi(psi),
        "zero_share_reference": round(zero_share_prior, 6),
        "zero_share_current": round(zero_share_recent, 6),
        "note": (
            "PSI compares distribution shapes only; it does not explain causes. "
            "A high PSI on low-volume series can reflect normal intermittency."
        ),
    }


def price_window_drift(price_stats: pd.DataFrame) -> dict:
    """PSI of per-series first vs last observed price (price-level drift)."""

    required = {"first_price", "last_price", "price_changes"}
    missing = required - set(price_stats.columns)
    if missing:
        raise ValueError(f"price_stats missing columns: {sorted(missing)}")

    first = price_stats["first_price"].to_numpy(dtype=float)
    last = price_stats["last_price"].to_numpy(dtype=float)

    # Only series priced in both windows can be compared.
    mask = np.isfinite(first) & np.isfinite(last)
    if mask.sum() < 100:
        raise ValueError("Fewer than 100 series have both first and last price")

    psi = population_stability_index(first[mask], last[mask])
    changed_share = float((price_stats.loc[mask, "price_changes"] > 0).mean())

    return {
        "metric": "first_vs_last_observed_price",
        "series_compared": int(mask.sum()),
        "psi": round(psi, 6),
        "classification": classify_psi(psi),
        "share_of_series_with_any_price_change": round(changed_share, 6),
        "note": (
            "Compares the first and last observed price per series across the "
            "whole history, not a controlled before/after experiment."
        ),
    }


def drift_report(
    tail: pd.DataFrame, price_stats: pd.DataFrame, window_days: int = 28
) -> dict:
    """Full drift report combining demand and price windows."""

    return {
        "demand": demand_window_drift(tail, window_days=window_days),
        "price": price_window_drift(price_stats),
        "psi_thresholds": {
            "stable_below": PSI_STABLE_BELOW,
            "moderate_below": PSI_MODERATE_BELOW,
            "note": "Rule-of-thumb thresholds, not statistical tests.",
        },
    }
