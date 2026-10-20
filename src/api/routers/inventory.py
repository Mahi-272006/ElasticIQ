"""Inventory recommendation endpoint.

The M5 dataset contains no stock positions or lead times: every inventory
parameter is user-supplied, and demand statistics come either explicitly or
from the trained forecast for a series. Results are recommendations under
stated assumptions, never a reflection of real warehouse state.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.routers.catalog import get_state
from src.api.schemas import InventoryRequestBody, InventoryResponse
from src.api.services import AppState
from src.db.models import InventoryRecommendation as InventoryRecommendationRow
from src.db.session import get_db
from src.inventory.planning import InventoryInputs, recommend

router = APIRouter(prefix="/api/v1/inventory", tags=["inventory"])


@router.post("/recommendations", response_model=InventoryResponse)
def create_recommendation(
    body: InventoryRequestBody,
    state: Annotated[AppState, Depends(get_state)],
    db: Annotated[Session, Depends(get_db)],
) -> InventoryResponse:
    """Compute an inventory plan from forecast or explicit demand stats."""

    try:
        inputs = InventoryInputs(**body.inputs)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    warnings: list[str] = []

    if body.daily_demand_mean is not None and body.daily_demand_std is not None:
        demand_source = "explicit"
        daily_mean = body.daily_demand_mean
        daily_std = body.daily_demand_std
        forecast_total = daily_mean * body.horizon_days
        warnings.append(
            "Demand statistics were supplied explicitly by the caller; they "
            "are not derived from the model or from M5 inventory data."
        )
    else:
        demand_source = "series_forecast"
        try:
            stats = state.demand_stats(body.series_id, body.horizon_days)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        daily_mean = stats["daily_mean"]
        daily_std = stats["daily_std"]
        forecast_total = stats["forecast_total_units"]

    result = recommend(
        daily_mean=daily_mean,
        daily_std=daily_std,
        horizon_days=body.horizon_days,
        forecast_demand_horizon=forecast_total,
        inputs=inputs,
    )

    recommendation_id: int | None = None
    if body.persist:
        row = InventoryRecommendationRow(
            series_id=body.series_id,
            horizon_days=body.horizon_days,
            lead_time_days=inputs.lead_time_days,
            service_level=inputs.service_level,
            assumed_on_hand=inputs.on_hand_units,
            results_json=result.model_dump_json(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        recommendation_id = row.id

    return InventoryResponse(
        recommendation_id=recommendation_id,
        series_id=body.series_id,
        demand_source=demand_source,
        recommendation=result.model_dump(),
        warnings=warnings,
    )
