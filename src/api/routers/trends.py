"""Sales trend endpoints over full-catalog daily aggregates.

Limitation stated honestly: processed aggregates exist per store and per
category separately (not per store x category slice), so ``store_id`` and
``cat_id`` cannot be combined in one request.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.schemas import TrendsResponse
from src.api.services import AppState, iso_date

router = APIRouter(prefix="/api/v1/sales", tags=["sales"])


def get_state(request: Request) -> AppState:
    return request.app.state.elasticiq


@router.get("/trends", response_model=TrendsResponse)
def sales_trends(
    state: Annotated[AppState, Depends(get_state)],
    days: Annotated[int, Query(ge=1, le=1913)] = 90,
    store_id: Annotated[str | None, Query()] = None,
    cat_id: Annotated[str | None, Query()] = None,
) -> TrendsResponse:
    """Daily units for the last ``days`` days, optionally filtered."""

    if store_id is not None and cat_id is not None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Combining store_id and cat_id is not supported: processed "
                "aggregates are stored per store and per category separately. "
                "Filter by one dimension at a time."
            ),
        )

    if store_id is not None:
        frame = state.daily_store
        known = frame["store_id"].unique()
        if store_id not in known:
            raise HTTPException(status_code=404, detail=f"Unknown store_id: {store_id}")
        frame = frame[frame["store_id"] == store_id]
        source = "daily_store_units"
    elif cat_id is not None:
        frame = state.daily_category
        known = frame["cat_id"].unique()
        if cat_id not in known:
            raise HTTPException(status_code=404, detail=f"Unknown cat_id: {cat_id}")
        frame = frame[frame["cat_id"] == cat_id]
        source = "daily_category_units"
    else:
        frame = state.daily_total
        source = "daily_total_units"

    recent = frame.sort_values("date").tail(days)

    points = [
        {
            "date": iso_date(record["date"]),
            "d": record["d"],
            "units": float(record["units"]),
        }
        for record in recent.to_dict("records")
    ]

    return TrendsResponse(
        source_aggregate=source,
        days=days,
        store_id=store_id,
        cat_id=cat_id,
        total_units=round(sum(point["units"] for point in points), 2),
        points=points,
    )
