"""Constrained pricing optimization over an explicit objective.

Objective (stated, not hidden): maximize expected gross profit
``units(price) * (price - unit_cost)`` over a grid of candidate prices,
subject to the user's :class:`~src.pricing.scenario.PricingConstraints`.

A price is only reported as "optimal under stated assumptions" — the
optimality is relative to the demand model and constraints supplied by the
caller, never a claim about real-world causal optimality.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from src.pricing.scenario import (
    PricingScenarioRequest,
    price_is_permitted,
    projected_units,
    validate_optimizer_inputs,
)

GRID_SIZE = 200


class OptimizeRequest(BaseModel):
    """Optimization input: one series, a demand assumption, and constraints."""

    current_price: float = Field(gt=0)
    unit_cost: float = Field(ge=0)
    baseline_units: float = Field(
        ge=0, description="Forecast demand at current price over the horizon"
    )
    demand_model: str = "elasticity"
    elasticity: float | None = None
    linear_slope: float | None = None
    horizon_days: int = Field(default=28, ge=1, le=92)
    constraints: dict

    grid_size: int = Field(default=GRID_SIZE, ge=10, le=2000)


class OptimizeResult(BaseModel):
    """Result of the constrained grid search."""

    optimal_price: float
    optimal_units: float
    optimal_revenue: float
    optimal_gross_profit: float
    current_gross_profit: float
    improvement_percent: float
    evaluated_prices: int
    feasible_prices: int
    objective: str
    assumptions: list[str]
    all_prices: list[dict] = Field(
        default_factory=list, description="Price/profit curve (sampled)"
    )


def optimize_price(request: OptimizeRequest) -> OptimizeResult:
    """Grid-search candidate prices to maximize expected gross profit."""

    from src.pricing.scenario import PricingConstraints

    constraints = PricingConstraints(**request.constraints)

    scenario = PricingScenarioRequest(
        current_price=request.current_price,
        candidate_price=request.current_price,  # placeholder; overridden below
        unit_cost=request.unit_cost,
        baseline_units=request.baseline_units,
        demand_model=request.demand_model,  # type: ignore[arg-type]
        elasticity=request.elasticity,
        linear_slope=request.linear_slope,
        assumption_source="user",
        horizon_days=request.horizon_days,
        constraints=constraints,
    )

    low = max(constraints.min_price, request.current_price * 0.5)
    high = min(constraints.max_price, request.current_price * 1.5)
    if high <= low:
        low, high = constraints.min_price, constraints.max_price

    candidates = np.linspace(low, high, request.grid_size)
    validate_optimizer_inputs(candidates, scenario)

    best: dict | None = None
    curve: list[dict] = []
    feasible_count = 0

    for price in candidates:
        price = float(round(price, 4))
        scenario.candidate_price = price

        if price_is_permitted(price, scenario):
            continue
        feasible_count += 1

        units, _ = projected_units(scenario)
        units = min(units, constraints.inventory_cap_units or units)
        profit = units * (price - request.unit_cost)
        revenue = units * price

        curve.append(
            {
                "price": price,
                "units": round(units, 4),
                "revenue": round(revenue, 4),
                "gross_profit": round(profit, 4),
            }
        )

        if best is None or profit > best["profit"]:
            best = {
                "price": price,
                "units": units,
                "revenue": revenue,
                "profit": profit,
            }

    if best is None:
        raise ValueError(
            "No feasible price under the given constraints; relax constraints "
            "or widen the price bounds."
        )

    current_units, _ = projected_units(scenario)
    current_units = min(current_units, constraints.inventory_cap_units or current_units)
    current_profit = current_units * (request.current_price - request.unit_cost)

    improvement = (
        (best["profit"] - current_profit) / current_profit * 100
        if current_profit > 0
        else 0.0
    )

    # Sample the curve to a manageable size for API responses.
    step = max(1, len(curve) // 50)
    sampled = curve[::step]

    assumptions = [
        f"Demand model: {request.demand_model}",
    ]
    if request.demand_model == "elasticity":
        assumptions.append(f"Assumed elasticity: {request.elasticity}")
    if request.demand_model == "linear_slope":
        assumptions.append(f"Assumed linear slope: {request.linear_slope}")
    assumptions.append(
        "Optimality is relative to the stated demand model and constraints; "
        "it is not a causal claim about real-world pricing."
    )

    return OptimizeResult(
        optimal_price=best["price"],
        optimal_units=round(best["units"], 4),
        optimal_revenue=round(best["revenue"], 4),
        optimal_gross_profit=round(best["profit"], 4),
        current_gross_profit=round(current_profit, 4),
        improvement_percent=round(improvement, 4),
        evaluated_prices=request.grid_size,
        feasible_prices=feasible_count,
        objective="maximize expected gross profit = units(price) * (price - unit_cost)",
        assumptions=assumptions,
        all_prices=sampled,
    )
