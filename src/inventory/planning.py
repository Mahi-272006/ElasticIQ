"""Inventory planning calculations.

What this module computes
-------------------------
Estimates from *forecast demand* plus user-supplied inventory parameters:

- lead-time demand (expected demand during supplier lead time)
- safety stock for a target service level (normal approximation)
- reorder point (lead-time demand + safety stock)
- suggested replenishment quantity (order-up-to level minus on-hand)
- stockout / overstock indicators

What this module does NOT claim
-------------------------------
The M5 dataset contains no stock-on-hand, open-orders, or lead-time data.
Every inventory parameter (on-hand quantity, lead time, service level,
review period) must be supplied by the user; results are *recommendations
under stated assumptions*, never a reflection of actual warehouse state.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

# Standard normal quantiles for common service levels (two-sided safety).
SERVICE_LEVEL_Z: dict[float, float] = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.98: 2.0537,
    0.99: 2.3263,
    0.999: 3.0902,
}

Status = Literal["ok", "stockout_risk", "overstock_risk"]


class InventoryInputs(BaseModel):
    """User-supplied inventory parameters for one product-store series."""

    service_level: float = Field(
        default=0.95, gt=0.5, lt=1.0, description="Target probability of no stockout"
    )
    lead_time_days: int = Field(
        default=7, ge=1, le=92, description="Supplier lead time in days"
    )
    review_period_days: int = Field(
        default=7, ge=1, le=92, description="Ordering cadence in days"
    )
    on_hand_units: float = Field(
        default=0, ge=0, description="Actual stock on hand (user-provided)"
    )
    open_order_units: float = Field(
        default=0, ge=0, description="Units already ordered, arriving before lead time"
    )
    overstock_months_threshold: float = Field(
        default=6.0, gt=0, description="Months of supply considered overstocked"
    )


class InventoryRecommendation(BaseModel):
    """Estimated inventory plan under stated assumptions."""

    horizon_days: int
    forecast_demand_horizon: float
    daily_demand_mean: float
    daily_demand_std: float

    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    order_up_to_level: float
    suggested_replenishment: float

    expected_days_of_cover: float | None
    status: Status
    status_reasons: list[str]

    assumptions: list[str]
    warnings: list[str]


def safety_stock(daily_std: float, service_level: float, lead_time_days: int) -> float:
    """Safety stock = z(service level) * demand std during lead time.

    Uses the normal approximation with the demand variance assumed
    independent across days.
    """

    z = _z_for_service_level(service_level)
    return z * daily_std * np.sqrt(lead_time_days)


def _z_for_service_level(service_level: float) -> float:
    """Return z for a service level, interpolating the table when needed."""

    for level, z in SERVICE_LEVEL_Z.items():
        if abs(level - service_level) < 1e-9:
            return z

    # Rational approximation of the normal quantile (Acklam's simple form).
    from scipy.stats import norm

    return float(norm.ppf(service_level))


def reorder_point(
    daily_mean: float, daily_std: float, inputs: InventoryInputs
) -> float:
    """Reorder point = expected lead-time demand + safety stock."""

    lead_demand = daily_mean * inputs.lead_time_days
    safety = safety_stock(daily_std, inputs.service_level, inputs.lead_time_days)
    return lead_demand + safety


def coverage_period_demand(daily_mean: float, inputs: InventoryInputs) -> float:
    """Expected demand during lead time plus one review period."""

    period = inputs.lead_time_days + inputs.review_period_days
    return daily_mean * period


def recommend(
    daily_mean: float,
    daily_std: float,
    horizon_days: int,
    forecast_demand_horizon: float,
    inputs: InventoryInputs,
) -> InventoryRecommendation:
    """Compute the full inventory recommendation for one series."""

    warnings: list[str] = []
    status_reasons: list[str] = []

    if daily_mean <= 0:
        warnings.append(
            "Forecast demand is zero; recommendation degenerates to "
            "'no replenishment needed' unless a demand shift is expected."
        )

    if not any(abs(inputs.service_level - level) < 1e-9 for level in SERVICE_LEVEL_Z):
        warnings.append(
            f"Service level {inputs.service_level} is not in the standard table; "
            "z interpolated from the normal distribution."
        )

    lead_demand = daily_mean * inputs.lead_time_days
    safety = safety_stock(daily_std, inputs.service_level, inputs.lead_time_days)
    rop = lead_demand + safety

    coverage_demand = coverage_period_demand(daily_mean, inputs)
    order_up_to = coverage_demand + safety

    position = inputs.on_hand_units + inputs.open_order_units
    replenishment = max(order_up_to - position, 0.0)

    days_of_cover = position / daily_mean if daily_mean > 0 else None

    status: Status = "ok"
    if position < rop:
        status = "stockout_risk"
        status_reasons.append(
            f"Inventory position {position:.0f} is below the reorder point "
            f"{rop:.0f}; a stockout is likely within the lead time."
        )
    elif days_of_cover is not None and (
        days_of_cover > inputs.overstock_months_threshold * 30
    ):
        status = "overstock_risk"
        status_reasons.append(
            f"Position covers {days_of_cover:.0f} days "
            f"(> {inputs.overstock_months_threshold} months); carrying-cost risk."
        )

    assumptions = [
        "Demand statistics come from the forecast/horizon supplied by the caller.",
        "Lead time and review period are fixed and known (user-supplied).",
        "Daily demand treated as independent with a normal lead-time aggregate "
        "(classic safety-stock approximation).",
        "On-hand and open-order quantities are user-supplied, not from M5 data.",
    ]

    return InventoryRecommendation(
        horizon_days=horizon_days,
        forecast_demand_horizon=round(forecast_demand_horizon, 4),
        daily_demand_mean=round(daily_mean, 4),
        daily_demand_std=round(daily_std, 4),
        lead_time_demand=round(lead_demand, 4),
        safety_stock=round(safety, 4),
        reorder_point=round(rop, 4),
        order_up_to_level=round(order_up_to, 4),
        suggested_replenishment=round(replenishment, 4),
        expected_days_of_cover=round(days_of_cover, 1) if days_of_cover else None,
        status=status,
        status_reasons=status_reasons,
        assumptions=assumptions,
        warnings=warnings,
    )
