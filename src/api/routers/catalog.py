"""Catalog endpoints: products (paginated), product detail, stores.

Data source: processed Parquet derived from the real M5 dataset
(``products``, ``price_stats``, ``series_history_tail``).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.schemas import (
    ProductDetail,
    ProductOut,
    ProductsPage,
    StoreOut,
    StoresResponse,
)
from src.api.services import AppState, iso_date

router = APIRouter(prefix="/api/v1", tags=["catalog"])


def get_state(request: Request) -> AppState:
    """FastAPI dependency exposing the shared application state."""

    return request.app.state.elasticiq


@router.get("/products", response_model=ProductsPage, response_model_by_alias=False)
def list_products(
    state: Annotated[AppState, Depends(get_state)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    cat_id: Annotated[str | None, Query()] = None,
    store_id: Annotated[str | None, Query()] = None,
    search: Annotated[str | None, Query(max_length=64)] = None,
) -> ProductsPage:
    """Paginated product catalog with optional filters."""

    frame = state.products

    if cat_id is not None:
        frame = frame[frame["cat_id"] == cat_id]
    if store_id is not None:
        frame = frame[frame["store_id"] == store_id]
    if search is not None:
        needle = search.lower()
        mask = frame["id"].str.lower().str.contains(needle, regex=False) | frame[
            "item_id"
        ].str.lower().str.contains(needle, regex=False)
        frame = frame[mask]

    total = len(frame)
    page = frame.iloc[offset : offset + limit]

    return ProductsPage(
        total=total,
        limit=limit,
        offset=offset,
        items=[ProductOut.model_validate(row) for row in page.to_dict("records")],
    )


@router.get(
    "/products/{series_id}", response_model=ProductDetail, response_model_by_alias=False
)
def get_product(
    series_id: str, state: Annotated[AppState, Depends(get_state)]
) -> ProductDetail:
    """One series with price statistics and its observed history."""

    row = state.product_row(series_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown series_id: {series_id}")

    stats_rows = state.price_stats[
        (state.price_stats["store_id"] == row["store_id"])
        & (state.price_stats["item_id"] == row["item_id"])
    ]
    price_stats: dict | None = None
    if not stats_rows.empty:
        raw = stats_rows.iloc[0].to_dict()
        price_stats = {
            key: (float(value) if hasattr(value, "item") else value)
            for key, value in raw.items()
        }

    history = state.product_history(series_id)
    history_payload = [
        {
            "date": str(record["date"]),
            "d": record["d"],
            "units": float(record["units"]),
        }
        for record in history.to_dict("records")
    ]

    return ProductDetail(
        product=ProductOut.model_validate(row.to_dict()),
        price_stats=price_stats,
        history=history_payload,
        zero_demand_share=state.zero_demand_share(series_id),
    )


@router.get("/stores", response_model=StoresResponse)
def list_stores(state: Annotated[AppState, Depends(get_state)]) -> StoresResponse:
    """All stores with total units from the full-catalog aggregates."""

    daily = state.daily_store
    totals = daily.groupby("store_id")["units"].sum().sort_index()

    overall = float(totals.sum())
    first_date = iso_date(daily["date"].min())
    last_date = iso_date(daily["date"].max())

    state_map = state.products.drop_duplicates("store_id").set_index("store_id")[
        "state_id"
    ]

    items = [
        StoreOut(
            store_id=store_id,
            state_id=str(state_map.get(store_id)) if store_id in state_map else None,
            total_units=round(float(units), 2),
            share_percent=round(float(units) / overall * 100, 4) if overall else 0.0,
            first_date=first_date,
            last_date=last_date,
        )
        for store_id, units in totals.items()
    ]

    return StoresResponse(total=len(items), items=items)
