"""Business overview for the landing dashboard (real data only)."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.routers.catalog import get_state
from src.api.schemas import CoverageBlock, MetricShare, ModelSummary, OverviewResponse
from src.api.services import AppState, iso_date

router = APIRouter(prefix="/api/v1", tags=["overview"])


def _model_summary(state: AppState) -> ModelSummary | None:
    """Build the model block from the stored evaluation artifact."""

    try:
        version = state.latest_model_version()
    except FileNotFoundError:
        return None

    evaluation_path = state.models_dir / version / "evaluation.json"
    if not evaluation_path.is_file():
        return ModelSummary(version=version)

    with evaluation_path.open(encoding="utf-8") as handle:
        evaluation = json.load(handle)

    model_block = evaluation.get("model", {})
    wape_by_horizon = {
        horizon: float(metrics.get("wape_percent"))
        for horizon, metrics in model_block.get("by_horizon", {}).items()
    }
    baselines = {
        name: payload.get("overall", {}).get("wape_percent")
        for name, payload in evaluation.get("baselines", {}).items()
    }

    return ModelSummary(
        version=version,
        wape_percent_by_horizon=wape_by_horizon,
        baseline_comparison={
            "model_overall_wape_percent": model_block.get("overall", {}).get(
                "wape_percent"
            ),
            "baselines_overall_wape_percent": baselines,
        },
    )


@router.get("/overview", response_model=OverviewResponse)
def overview(state: Annotated[AppState, Depends(get_state)]) -> OverviewResponse:
    """Coverage, totals, category/store shares, and model performance."""

    daily_total = state.daily_total
    products = state.products

    total_units = float(daily_total["units"].sum())
    days = len(daily_total)

    category_units = state.daily_category.groupby("cat_id")["units"].sum()
    categories = [
        MetricShare(
            key=str(key),
            units=round(float(value), 2),
            share_percent=round(float(value) / total_units * 100, 4),
        )
        for key, value in category_units.sort_values(ascending=False).items()
    ]

    store_units = state.daily_store.groupby("store_id")["units"].sum()
    stores = [
        MetricShare(
            key=str(key),
            units=round(float(value), 2),
            share_percent=round(float(value) / total_units * 100, 4),
        )
        for key, value in store_units.sort_values(ascending=False).items()
    ]

    coverage = CoverageBlock(
        first_date=iso_date(daily_total["date"].min()),
        last_date=iso_date(daily_total["date"].max()),
        days=days,
        series_in_catalog=int(len(products)),
        stores=int(products["store_id"].nunique()),
        categories=int(products["cat_id"].nunique()),
    )

    return OverviewResponse(
        coverage=coverage,
        totals={
            "total_units": round(total_units, 2),
            "mean_daily_units": round(total_units / days, 2),
            "last_day_units": round(float(daily_total["units"].iloc[-1]), 2),
        },
        categories=categories,
        stores=stores,
        model=_model_summary(state),
        data_notes=[
            "Aggregates cover all series in the catalog (full M5 slice), "
            "computed by the offline pipeline from the raw CSVs.",
            "Model metrics are holdout WAPE on chronological validation days "
            "and are compared against naive baselines on identical inputs.",
        ],
    )
