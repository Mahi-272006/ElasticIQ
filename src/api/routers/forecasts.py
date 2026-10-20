"""Forecast endpoints: create a run (real model inference) and read it back.

Runs are persisted to the database (ForecastRun + ForecastPoint) so an id can
be shared and audited later. Inference itself uses the trained bundle and the
stored per-series history tail; future prices are assumed equal to the last
observed price (stated in the response).
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from src.api.routers.catalog import get_state
from src.api.schemas import (
    EVALUATED_HORIZONS,
    ForecastPointOut,
    ForecastRequest,
    ForecastResponse,
    ForecastRunsPage,
    ForecastRunSummary,
)
from src.api.services import AppState, iso_date
from src.db.models import ForecastPoint, ForecastRun
from src.db.session import get_db

router = APIRouter(prefix="/api/v1/forecasts", tags=["forecasts"])

ASSUMPTIONS = [
    "Future sell prices are assumed equal to the last observed price "
    "(no planned price changes supplied).",
    "Predictions are fed back recursively into the feature history "
    "(no ground-truth future sales are used).",
    "SNAP flags and calendar events are known in advance and used as-is.",
    "Metrics were measured on a chronological holdout; this run makes no "
    "claim about accuracy for these specific future days.",
]


@router.post("", response_model=ForecastResponse, status_code=201)
def create_forecast(
    body: ForecastRequest,
    state: Annotated[AppState, Depends(get_state)],
    db: Annotated[Session, Depends(get_db)],
) -> ForecastResponse:
    """Forecast one series with the trained model and persist the run."""

    if state.product_row(body.series_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown series_id: {body.series_id}"
        )

    try:
        predictions = state.forecast_series(body.series_id, body.horizon_days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    bundle = state.bundle
    calendar = state.calendar
    origin_day = int(predictions["day_idx"].min()) - 1
    origin_date = iso_date(
        calendar.loc[calendar["day_idx"] == origin_day, "date"].iloc[0]
    )

    points = [
        ForecastPointOut(
            day_idx=int(record["day_idx"]),
            d=str(record["d"]),
            date=iso_date(record["date"]),
            predicted_units=round(float(record["predicted_units"]), 4),
        )
        for record in predictions.to_dict("records")
    ]

    response = ForecastResponse(
        forecast_id=None,
        series_id=body.series_id,
        horizon_days=body.horizon_days,
        model_version=str(bundle.get("version", "unknown")),
        origin_day=origin_day,
        origin_date=origin_date,
        beyond_evaluated_horizons=body.horizon_days not in EVALUATED_HORIZONS,
        assumptions=list(ASSUMPTIONS),
        points=points,
    )

    if body.persist:
        run = ForecastRun(
            series_id=body.series_id,
            horizon_days=body.horizon_days,
            model_version=response.model_version,
            origin_day=origin_day,
            origin_date=origin_date,
            assumptions_json=json.dumps(response.assumptions),
            points=[
                ForecastPoint(
                    day_idx=point.day_idx,
                    d=point.d,
                    date=point.date,
                    predicted_units=point.predicted_units,
                )
                for point in points
            ],
        )
        db.add(run)
        db.commit()
        db.refresh(run)

        response.forecast_id = run.id
        response.created_at = run.created_at

    return response


@router.get("", response_model=ForecastRunsPage)
def list_forecasts(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    series_id: Annotated[str | None, Query()] = None,
) -> ForecastRunsPage:
    """Most recent stored forecast runs (headers only)."""

    query = select(ForecastRun)
    count_query = select(func.count(ForecastRun.id))
    if series_id is not None:
        query = query.where(ForecastRun.series_id == series_id)
        count_query = count_query.where(ForecastRun.series_id == series_id)

    total = int(db.scalar(count_query) or 0)
    runs = db.scalars(
        query.order_by(ForecastRun.created_at.desc()).offset(offset).limit(limit)
    ).all()

    items = [
        ForecastRunSummary(
            forecast_id=run.id,
            series_id=run.series_id,
            horizon_days=run.horizon_days,
            model_version=run.model_version,
            origin_day=run.origin_day,
            origin_date=run.origin_date,
            created_at=run.created_at,
        )
        for run in runs
    ]

    return ForecastRunsPage(total=total, limit=limit, offset=offset, items=items)


@router.get("/{forecast_id}", response_model=ForecastResponse)
def get_forecast(
    forecast_id: str, db: Annotated[Session, Depends(get_db)]
) -> ForecastResponse:
    """Read back a stored forecast run with its points."""

    run = (
        db.scalars(
            select(ForecastRun)
            .options(joinedload(ForecastRun.points))
            .where(ForecastRun.id == forecast_id)
        )
        .unique()
        .one_or_none()
    )

    if run is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown forecast_id: {forecast_id}"
        )

    return ForecastResponse(
        forecast_id=run.id,
        series_id=run.series_id,
        horizon_days=run.horizon_days,
        model_version=run.model_version,
        origin_day=run.origin_day,
        origin_date=run.origin_date,
        beyond_evaluated_horizons=run.horizon_days not in EVALUATED_HORIZONS,
        assumptions=json.loads(run.assumptions_json),
        points=[
            ForecastPointOut(
                day_idx=point.day_idx,
                d=point.d,
                date=point.date,
                predicted_units=point.predicted_units,
            )
            for point in sorted(run.points, key=lambda p: p.day_idx)
        ],
        created_at=run.created_at,
    )
