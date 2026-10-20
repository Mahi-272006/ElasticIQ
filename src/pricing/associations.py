"""Observational price-demand association analysis (NOT causal).

The M5 data is observational. Prices co-move with promotions, SNAP windows,
calendar periods, and store strategy. Any regression coefficient or
correlation computed here describes *association only*; it must never be
presented as the causal effect of a price change.

This module provides:

1. Descriptive price variation summaries per series (already produced by
   the data pipeline as ``price_stats``).
2. Within-series price-demand associations from the sample panel using a
   log-log fixed-effects style regression (series means absorbed), with
   explicit diagnostics (overlap, confounding flags, sensitivity).
3. A clearly labeled *simulated* dataset generator with a known causal
   price effect, used only to demonstrate causal-method plumbing
   (instrumental-variables style sanity check). Simulated results are
   never mixed with real M5 results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.pricing.scenario import ILLUSTRATIVE_ELASTICITY

SIMULATED_DATASET_LABEL = "SIMULATED — synthetic data, not M5"

# Confounders that plausibly drive both price and demand in M5.
CONFOUNDER_NOTE = [
    "promotions are not flagged in M5; price drops often ARE promotions",
    "SNAP assistance days change both basket size and store traffic",
    "seasonality: both price and demand trend within the year",
    "store-level pricing strategy (EDLP vs HiLo) differs by store",
]


@dataclass
class AssociationResult:
    """Log-log price-demand association for one series (or pooled)."""

    slope: float
    intercept: float
    n_observations: int
    price_variation_ratio: float
    r_squared: float
    standard_error: float | None = None
    diagnostics: list[str] = field(default_factory=list)
    interpretation: str = ""


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Ordinary least squares with a constant term.

    Returns (intercept, slope, r_squared, slope_se).
    """

    design = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residuals = y - fitted

    n, k = design.shape
    ss_res = float((residuals**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    dof = max(n - k, 1)
    sigma2 = ss_res / dof
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    slope_se = float(np.sqrt(covariance[1, 1])) if n > 2 else None

    return beta[0], beta[1], r_squared, slope_se


def within_series_association(
    units: np.ndarray,
    prices: np.ndarray,
    min_distinct_prices: int = 3,
) -> AssociationResult:
    """Log-log association between price and units for one series.

    Fits log(units + 1) = a + b * log(price) + e on days with a known price.
    The slope ``b`` is an *association*; the constant-elasticity model it
    suggests is an assumption, not an estimated causal effect.
    """

    diagnostics: list[str] = []

    mask = np.isfinite(prices) & np.isfinite(units) & (prices > 0)
    x = np.log(prices[mask])
    y = np.log(units[mask] + 1.0)

    distinct_prices = len(np.unique(np.round(prices[mask], 4)))
    if distinct_prices < min_distinct_prices:
        return AssociationResult(
            slope=float("nan"),
            intercept=float("nan"),
            n_observations=int(mask.sum()),
            price_variation_ratio=float("nan"),
            r_squared=float("nan"),
            diagnostics=[
                f"only {distinct_prices} distinct price(s); "
                "association not identifiable"
            ],
            interpretation="Insufficient price variation to say anything about "
            "price-demand association.",
        )

    price_ratio = float(prices[mask].max() / prices[mask].min())
    if price_ratio < 1.1:
        diagnostics.append(
            f"price range narrow (max/min={price_ratio:.2f}); slope unstable"
        )

    intercept, slope, r_squared, slope_se = _ols(np.asarray(x), np.asarray(y))

    diagnostics.append("ASSOCIATION ONLY: not a causal estimate; see CONFOUNDER_NOTE")

    return AssociationResult(
        slope=float(slope),
        intercept=float(intercept),
        n_observations=int(mask.sum()),
        price_variation_ratio=round(price_ratio, 4),
        r_squared=round(float(r_squared), 4),
        standard_error=round(float(slope_se), 4) if slope_se is not None else None,
        diagnostics=diagnostics,
        interpretation=(
            f"A 1% higher price is associated with a ~{abs(slope) * 100:.1f}% "
            f"{'lower' if slope < 0 else 'higher'} demand in this series. "
            "This is an observational association over historical data."
        ),
    )


def pooled_association(panel: pd.DataFrame) -> AssociationResult:
    """Association across all priced panel rows with series fixed effects.

    Log(units + 1) = series_mean + b * log(price) + e. Series means absorb
    time-invariant differences between products/stores; time-varying
    confounders (promotions, SNAP, seasonality) still bias the estimate.
    """

    frame = panel.dropna(subset=["sell_price"]).copy()
    frame = frame[frame["sell_price"] > 0]

    y = np.log(frame["units"].to_numpy(dtype=np.float64) + 1.0)
    x = np.log(frame["sell_price"].to_numpy(dtype=np.float64))

    # Series fixed effects via demeaning within series.
    series_means_y = (
        y
        - pd.Series(y, index=frame["id"])
        .groupby(frame["id"])
        .transform("mean")
        .to_numpy()
    )
    series_means_x = (
        x
        - pd.Series(x, index=frame["id"])
        .groupby(frame["id"])
        .transform("mean")
        .to_numpy()
    )

    slope = float((series_means_x @ series_means_y) / (series_means_x @ series_means_x))
    r_squared = 1.0 - float(
        ((series_means_y - slope * series_means_x) ** 2).sum()
        / (series_means_y**2).sum()
    )

    diagnostics = [
        "Series fixed effects absorb time-invariant product/store differences.",
        "ASSOCIATION ONLY: promotions and SNAP are unobserved confounders here.",
    ]

    return AssociationResult(
        slope=round(slope, 4),
        intercept=0.0,
        n_observations=int(len(frame)),
        price_variation_ratio=round(
            float(frame["sell_price"].max() / frame["sell_price"].min()), 4
        ),
        r_squared=round(r_squared, 4),
        diagnostics=diagnostics,
        interpretation=(
            f"Pooled within-series slope {slope:.3f}: a 1% higher price is "
            "associated with a ~"
            f"{abs(slope) * 100:.1f}% demand difference within the same "
            "series. Confounded; do not treat as an elasticity estimate."
        ),
    )


def sensitivity_range(panel: pd.DataFrame) -> dict:
    """Spread of per-series slopes, showing estimate instability."""

    slopes = []
    for _, group in panel.groupby("id", sort=False):
        result = within_series_association(
            group["units"].to_numpy(dtype=float),
            group["sell_price"].to_numpy(dtype=float),
        )
        if np.isfinite(result.slope):
            slopes.append(result.slope)

    if not slopes:
        return {"series_with_identifiable_slope": 0}

    array = np.array(slopes)
    return {
        "series_with_identifiable_slope": len(slopes),
        "slope_p10": round(float(np.quantile(array, 0.10)), 4),
        "slope_median": round(float(np.median(array)), 4),
        "slope_p90": round(float(np.quantile(array, 0.90)), 4),
        "share_negative": round(float((array < 0).mean()), 4),
        "note": (
            "The wide spread across series is itself evidence against a single "
            "causal elasticity; the pricing simulator therefore requires the "
            f"user to state an assumption (default illustrative value "
            f"{ILLUSTRATIVE_ELASTICITY})."
        ),
    }


# --------------------------------------------------- simulated demonstration


@dataclass
class SimulatedDataset:
    """Synthetic price-demand data with a KNOWN causal effect."""

    frame: pd.DataFrame
    true_elasticity: float
    label: str = SIMULATED_DATASET_LABEL


def generate_simulated_experiment(
    n_series: int = 20,
    n_days: int = 120,
    true_elasticity: float = -1.3,
    seed: int = 42,
    randomized: bool = True,
) -> SimulatedDataset:
    """Simulate price-demand data under a known constant elasticity.

    With ``randomized=True`` prices are drawn independently of demand
    shocks (an A/B-test-like design) so OLS recovers the true elasticity.
    With ``randomized=False`` a hidden promotion process cuts prices AND
    lifts demand, biasing the naive estimate — demonstrating confounding.
    """

    rng = np.random.default_rng(seed)
    rows = []

    for series in range(n_series):
        base_demand = rng.uniform(5, 50)
        baseline_price = rng.uniform(2, 10)
        promotion_strength = rng.uniform(0.3, 1.0)

        for day in range(n_days):
            if randomized:
                price = baseline_price * rng.uniform(0.7, 1.3)
                promoted = False
            else:
                promoted = rng.random() < 0.2
                price = baseline_price * (0.75 if promoted else 1.0)
                price *= rng.uniform(0.95, 1.05)

            demand_shock = rng.lognormal(0, 0.15)
            demand = (
                base_demand * (price / baseline_price) ** true_elasticity * demand_shock
            )
            if promoted and not randomized:
                # Hidden promotion effect on top of the price effect.
                demand *= 1.0 + promotion_strength

            rows.append(
                {
                    "series_id": f"sim_{series:03d}",
                    "day": day,
                    "price": round(float(price), 4),
                    "units": float(np.round(max(demand, 0.0), 2)),
                    "randomized": randomized,
                    "promoted": bool(promoted),
                }
            )

    return SimulatedDataset(frame=pd.DataFrame(rows), true_elasticity=true_elasticity)


def simulate_ols_estimate(data: SimulatedDataset) -> dict:
    """Recover the elasticity from simulated data via within-series OLS."""

    frame = data.frame
    estimates = []

    for _, group in frame.groupby("series_id"):
        x = np.log(group["price"].to_numpy(dtype=float))
        y = np.log(group["units"].to_numpy(dtype=float) + 1.0)
        demean_x = x - x.mean()
        demean_y = y - y.mean()
        estimates.append(float((demean_x @ demean_y) / (demean_x @ demean_x)))

    estimate = float(np.median(estimates))

    return {
        "label": data.label,
        "design": "randomized" if frame["randomized"].iloc[0] else "confounded",
        "true_elasticity": data.true_elasticity,
        "estimated_elasticity_median": round(estimate, 4),
        "bias": round(estimate - data.true_elasticity, 4),
        "note": (
            "On randomized data the estimate should be close to the truth; "
            "on confounded data it is biased away from it. This is why the "
            "M5 observational slope is not used as an elasticity."
        ),
    }
