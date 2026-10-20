"""Pricing scenario simulation with explicit demand assumptions.

Honesty contract
----------------
The M5 dataset is observational: prices were not randomized, and price
movements co-occur with promotions, calendar periods, SNAP windows, and
store strategies. Nothing in this module claims a causal price effect.

Every scenario must therefore carry an explicit demand-response assumption:

- ``constant``     demand does not change with price (pure arithmetic)
- ``elasticity``   constant-elasticity model: units ratio = (p_new/p_cur)^e
- ``linear_slope`` linear demand curve: units change by slope per currency unit

The chosen assumption, its parameters, and its source (user-provided vs.
illustrative default) are echoed back in every result so downstream users
cannot mistake an assumption for a measured effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

DemandModel = Literal["constant", "elasticity", "linear_slope"]

# Illustrative default only — never estimated from M5 data.
ILLUSTRATIVE_ELASTICITY = -1.5
ILLUSTRATIVE_LINEAR_SLOPE = -0.5

MIN_PRICE_FLOOR = 0.01


class PricingConstraints(BaseModel):
    """Business rules applied to scenarios and optimization."""

    min_price: float = Field(gt=0, description="Minimum permitted unit price")
    max_price: float = Field(gt=0, description="Maximum permitted unit price")
    max_change_percent: float | None = Field(
        default=None, ge=0, description="Max |price change| vs current price, percent"
    )
    min_margin_percent: float | None = Field(
        default=None, description="Minimum gross margin percent of price"
    )
    inventory_cap_units: float | None = Field(
        default=None, ge=0, description="Units that can actually be sold (stock cap)"
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> PricingConstraints:
        if self.max_price <= self.min_price:
            raise ValueError("max_price must exceed min_price")
        return self


class PricingScenarioRequest(BaseModel):
    """A single what-if price change for one product-store series."""

    current_price: float = Field(gt=0)
    candidate_price: float = Field(gt=0)
    unit_cost: float = Field(ge=0)
    baseline_units: float = Field(
        ge=0, description="Demand forecast at the current price (units over horizon)"
    )
    demand_model: DemandModel = "constant"
    elasticity: float | None = Field(
        default=None, description="Required when demand_model='elasticity' (negative)"
    )
    linear_slope: float | None = Field(
        default=None,
        description="Units lost per currency unit of price increase; required for "
        "'linear_slope' (negative values mean demand falls as price rises)",
    )
    assumption_source: Literal["user", "illustrative"] = "user"
    horizon_days: int = Field(ge=1, le=92)
    constraints: PricingConstraints


@dataclass
class ScenarioOutcome:
    """Computed metrics for one scenario (before and after the price change)."""

    current_revenue: float
    candidate_revenue: float
    current_gross_profit: float
    candidate_gross_profit: float
    expected_units_current: float
    expected_units_candidate: float
    revenue_change_percent: float
    profit_change_percent: float
    capped_units: float | None
    warnings: list[str] = field(default_factory=list)


def projected_units(
    request: PricingScenarioRequest,
) -> tuple[float, list[str]]:
    """Apply the stated demand model and return candidate-price units.

    Returns (units, warnings). The demand model changes only the *units*
    projection; no claim is made that the model is causally correct.
    """

    warnings: list[str] = []
    baseline = request.baseline_units

    if request.demand_model == "constant":
        return baseline, ["Demand held constant: no price response assumed."]

    if request.demand_model == "elasticity":
        if request.elasticity is None:
            raise ValueError(
                "elasticity parameter required for demand_model='elasticity'"
            )
        if request.elasticity > 0:
            warnings.append(
                "Positive elasticity assumed (demand rises with price); "
                "unusual for retail — verify this is intentional."
            )

        ratio = (request.candidate_price / request.current_price) ** request.elasticity
        return baseline * ratio, warnings

    # linear_slope
    if request.linear_slope is None:
        raise ValueError(
            "linear_slope parameter required for demand_model='linear_slope'"
        )

    delta = request.candidate_price - request.current_price
    units = baseline + request.linear_slope * delta

    if units < 0:
        warnings.append(
            "Linear demand model produced negative units; floored at 0. "
            "The linear assumption is unrealistic for large price changes."
        )
        units = 0.0

    return units, warnings


def price_is_permitted(candidate: float, request: PricingScenarioRequest) -> list[str]:
    """Return constraint-violation messages for a candidate price."""

    constraints = request.constraints
    violations: list[str] = []

    if candidate < MIN_PRICE_FLOOR:
        violations.append(f"price below absolute floor {MIN_PRICE_FLOOR}")
    if candidate < constraints.min_price:
        violations.append(f"price {candidate:.2f} below min_price")
    if candidate > constraints.max_price:
        violations.append(f"price {candidate:.2f} above max_price")

    if constraints.max_change_percent is not None:
        max_abs = request.current_price * constraints.max_change_percent / 100
        if abs(candidate - request.current_price) > max_abs + 1e-9:
            violations.append(
                f"price change exceeds max_change_percent "
                f"{constraints.max_change_percent}%"
            )

    if constraints.min_margin_percent is not None and candidate > 0:
        margin = (candidate - request.unit_cost) / candidate * 100
        if margin < constraints.min_margin_percent:
            violations.append(
                f"margin {margin:.1f}% below min_margin_percent "
                f"{constraints.min_margin_percent}%"
            )

    return violations


def simulate_scenario(request: PricingScenarioRequest) -> ScenarioOutcome:
    """Compute revenue/profit outcomes for one pricing scenario."""

    units_candidate, warnings = projected_units(request)
    units_current = request.baseline_units

    capped = request.constraints.inventory_cap_units
    capped_units = None
    if capped is not None:
        if units_candidate > capped:
            warnings.append(
                f"Expected units {units_candidate:.1f} exceed inventory cap "
                f"{capped:.0f}; revenue computed on capped demand."
            )
        units_candidate = min(units_candidate, capped)
        units_current = min(units_current, capped)
        capped_units = capped

    current_revenue = units_current * request.current_price
    candidate_revenue = units_candidate * request.candidate_price
    current_profit = units_current * (request.current_price - request.unit_cost)
    candidate_profit = units_candidate * (request.candidate_price - request.unit_cost)

    def _pct(new: float, old: float) -> float:
        if old == 0:
            return 0.0 if new == 0 else float("inf")
        return round((new - old) / old * 100, 4)

    return ScenarioOutcome(
        current_revenue=round(current_revenue, 4),
        candidate_revenue=round(candidate_revenue, 4),
        current_gross_profit=round(current_profit, 4),
        candidate_gross_profit=round(candidate_profit, 4),
        expected_units_current=round(units_current, 4),
        expected_units_candidate=round(units_candidate, 4),
        revenue_change_percent=_pct(candidate_revenue, current_revenue),
        profit_change_percent=_pct(candidate_profit, current_profit),
        capped_units=capped_units,
        warnings=warnings,
    )


def demand_sensitivity_check(
    baseline_units: float,
    current_price: float,
    candidate_price: float,
    unit_cost: float,
    constraint: PricingConstraints,
    elasticities: tuple[float, ...] = (-0.5, -1.0, -1.5, -2.0, -3.0),
) -> list[dict]:
    """Sweep several elasticity assumptions for one price move.

    Demonstrates how conclusions depend on the assumed demand response —
    a required honesty check when no credible elasticity is estimated.
    """

    results = []
    for elasticity in elasticities:
        request = PricingScenarioRequest(
            current_price=current_price,
            candidate_price=candidate_price,
            unit_cost=unit_cost,
            baseline_units=baseline_units,
            demand_model="elasticity",
            elasticity=elasticity,
            assumption_source="illustrative",
            horizon_days=28,
            constraints=constraint,
        )
        outcome = simulate_scenario(request)
        results.append(
            {
                "elasticity": elasticity,
                "expected_units": outcome.expected_units_candidate,
                "candidate_revenue": outcome.candidate_revenue,
                "candidate_gross_profit": outcome.candidate_gross_profit,
                "profit_change_percent": outcome.profit_change_percent,
            }
        )
    return results


def validate_optimizer_inputs(
    prices: np.ndarray,
    request: PricingScenarioRequest,
) -> None:
    """Shared input checks for the optimization search."""

    if len(prices) == 0:
        raise ValueError("No candidate prices supplied")
    if not np.all(np.isfinite(prices)):
        raise ValueError("Candidate prices must be finite")
    if request.demand_model == "elasticity" and request.elasticity is None:
        raise ValueError("elasticity required for demand_model='elasticity'")
    if request.demand_model == "linear_slope" and request.linear_slope is None:
        raise ValueError("linear_slope required for demand_model='linear_slope'")
