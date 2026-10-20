"""Pricing endpoints: what-if simulation, optimization, sensitivity sweep.

Every response echoes the demand assumption it used. The M5 data is
observational, so no endpoint here claims a causal price effect.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.schemas import (
    PricingOptimizeBody,
    PricingScenarioBody,
    PricingScenarioResponse,
    PricingScenariosPage,
    PricingSensitivityBody,
)
from src.db.models import PricingScenario
from src.db.session import get_db
from src.pricing.optimize import optimize_price
from src.pricing.scenario import (
    PricingConstraints,
    demand_sensitivity_check,
    simulate_scenario,
)

router = APIRouter(prefix="/api/v1/pricing", tags=["pricing"])


@router.post("/scenarios", response_model=PricingScenarioResponse, status_code=201)
def run_scenario(
    body: PricingScenarioBody,
    db: Annotated[Session, Depends(get_db)],
) -> PricingScenarioResponse:
    """Simulate one price change under the stated demand assumption."""

    try:
        outcome = simulate_scenario(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    outcome_dict = asdict(outcome)
    scenario_id: int | None = None

    if body.persist:
        row = PricingScenario(
            series_id=body.series_id,
            request_json=body.model_dump_json(),
            result_json=json.dumps(outcome_dict),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        scenario_id = row.id

    return PricingScenarioResponse(
        scenario_id=scenario_id,
        series_id=body.series_id,
        request=body.model_dump(exclude={"series_id", "persist"}),  # type: ignore[arg-type]
        outcome=outcome_dict,
    )


@router.get("/scenarios", response_model=PricingScenariosPage)
def list_scenarios(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    series_id: Annotated[str | None, Query()] = None,
) -> PricingScenariosPage:
    """Most recently stored pricing scenarios."""

    query = select(PricingScenario)
    count_query = select(func.count(PricingScenario.id))
    if series_id is not None:
        query = query.where(PricingScenario.series_id == series_id)
        count_query = count_query.where(PricingScenario.series_id == series_id)

    total = int(db.scalar(count_query) or 0)
    rows = db.scalars(
        query.order_by(PricingScenario.created_at.desc()).offset(offset).limit(limit)
    ).all()

    items = [
        {
            "scenario_id": row.id,
            "series_id": row.series_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "request": json.loads(row.request_json),
            "outcome": json.loads(row.result_json),
        }
        for row in rows
    ]

    return PricingScenariosPage(total=total, limit=limit, offset=offset, items=items)


@router.post("/optimize")
def optimize_pricing(body: PricingOptimizeBody) -> dict:
    """Grid-search the profit-maximizing price under stated assumptions."""

    try:
        result = optimize_price(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    payload = result.model_dump()
    payload["series_id"] = body.series_id
    return payload


@router.post("/sensitivity")
def sensitivity(body: PricingSensitivityBody) -> dict:
    """Sweep elasticity assumptions to expose conclusion fragility."""

    try:
        constraint = PricingConstraints(**body.constraints)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    results = demand_sensitivity_check(
        baseline_units=body.baseline_units,
        current_price=body.current_price,
        candidate_price=body.candidate_price,
        unit_cost=body.unit_cost,
        constraint=constraint,
        elasticities=body.elasticities,
    )
    return {
        "elasticity_sweep": results,
        "note": (
            "Each row is a different assumed demand response, not a different "
            "measured effect. The spread shows how sensitive the conclusion "
            "is to an assumption the data cannot identify."
        ),
    }
