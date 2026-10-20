"""Typed request/response schemas for the ElasticIQ API (v1).

Engine-level models (pricing scenarios, optimization, inventory inputs) are
reused directly so the API contract cannot drift from the tested engines.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.pricing.optimize import OptimizeRequest
from src.pricing.scenario import PricingScenarioRequest

# Horizons the model was actually evaluated on (chronological holdout).
EVALUATED_HORIZONS = (7, 14, 28)
# The calendar extends 56 days beyond the last observed day (d_1913 -> d_1969).
MAX_FORECAST_HORIZON = 56


class ErrorDetail(BaseModel):
    """Uniform error payload."""

    detail: str


class HealthResponse(BaseModel):
    """Liveness/readiness report. Never pretends resources exist."""

    status: Literal["ok", "degraded"]
    service: str = "elasticiq-api"
    api_version: str = "v1"
    database: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]


class CoverageBlock(BaseModel):
    """Dataset coverage facts."""

    first_date: str
    last_date: str
    days: int
    series_in_catalog: int
    stores: int
    categories: int


class MetricShare(BaseModel):
    """Units and share for one group."""

    key: str
    units: float
    share_percent: float


class ModelSummary(BaseModel):
    """Compact model performance summary for dashboards."""

    version: str
    evaluation_horizons: list[int] = list(EVALUATED_HORIZONS)
    wape_percent_by_horizon: dict[str, float] = {}
    baseline_comparison: dict[str, Any] = {}
    note: str = (
        "WAPE on the chronological holdout (d_1886-d_1913). Baselines are "
        "evaluated on identical series and dates."
    )


class OverviewResponse(BaseModel):
    """Top-level business overview for the landing dashboard."""

    coverage: CoverageBlock
    totals: dict[str, float]
    categories: list[MetricShare]
    stores: list[MetricShare]
    model: ModelSummary | None
    data_notes: list[str]


class ProductOut(BaseModel):
    """Catalog entry for one product-store series."""

    series_id: str = Field(alias="id")
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str

    model_config = {"populate_by_name": True}


class ProductsPage(BaseModel):
    """Paginated product listing."""

    total: int
    limit: int
    offset: int
    items: list[ProductOut]


class StoreOut(BaseModel):
    """Store-level summary from full-catalog aggregates."""

    store_id: str
    state_id: str | None = None
    total_units: float
    share_percent: float
    first_date: str
    last_date: str


class StoresResponse(BaseModel):
    """Store listing."""

    total: int
    items: list[StoreOut]


class ProductDetail(BaseModel):
    """Product detail with price statistics and observed history."""

    product: ProductOut
    price_stats: dict[str, Any] | None
    history: list[dict[str, Any]]
    zero_demand_share: float | None


class TrendsResponse(BaseModel):
    """Daily units time series under the requested filters."""

    source_aggregate: str
    days: int
    store_id: str | None
    cat_id: str | None
    total_units: float
    points: list[dict[str, Any]]


class ForecastRequest(BaseModel):
    """Forecast one product-store series with the trained model."""

    series_id: str = Field(min_length=1, description="Product-store series id")
    horizon_days: int = Field(
        default=28,
        ge=1,
        le=MAX_FORECAST_HORIZON,
        description=(
            "Days to forecast. 7/14/28 are the evaluated horizons; the "
            "calendar supports at most 56 days beyond the last observed day."
        ),
    )
    persist: bool = Field(
        default=True, description="Store the run in the database and return an id"
    )


class ForecastPointOut(BaseModel):
    """One predicted day."""

    day_idx: int
    d: str
    date: str
    predicted_units: float


class ForecastResponse(BaseModel):
    """Forecast result with explicit assumptions."""

    forecast_id: str | None
    series_id: str
    horizon_days: int
    model_version: str
    origin_day: int
    origin_date: str
    beyond_evaluated_horizons: bool
    assumptions: list[str]
    points: list[ForecastPointOut]
    created_at: datetime | None = None


class ForecastRunSummary(BaseModel):
    """Stored forecast run header (without points)."""

    forecast_id: str
    series_id: str
    horizon_days: int
    model_version: str
    origin_day: int
    origin_date: str
    created_at: datetime


class ForecastRunsPage(BaseModel):
    """Paginated forecast run listing."""

    total: int
    limit: int
    offset: int
    items: list[ForecastRunSummary]


class PricingScenarioBody(PricingScenarioRequest):
    """Pricing what-if: engine request plus optional catalog linkage."""

    series_id: str | None = Field(
        default=None, description="Optional product-store series for bookkeeping"
    )
    persist: bool = True


class PricingScenarioResponse(BaseModel):
    """Echo of the request, the computed outcome, and its row id."""

    scenario_id: int | None
    series_id: str | None
    request: PricingScenarioRequest
    outcome: dict[str, Any]
    honesty_note: str = (
        "The M5 dataset is observational; outcomes follow the demand "
        "assumption you supplied, not a measured causal price effect."
    )


class PricingScenariosPage(BaseModel):
    """Paginated stored scenarios."""

    total: int
    limit: int
    offset: int
    items: list[dict[str, Any]]


class PricingOptimizeBody(OptimizeRequest):
    """Optimization request with optional catalog linkage."""

    series_id: str | None = None


class PricingSensitivityBody(BaseModel):
    """Sweep several elasticity assumptions for one price move."""

    baseline_units: float = Field(ge=0)
    current_price: float = Field(gt=0)
    candidate_price: float = Field(gt=0)
    unit_cost: float = Field(ge=0)
    constraints: dict
    elasticities: tuple[float, ...] = (-0.5, -1.0, -1.5, -2.0, -3.0)


class InventoryRequestBody(BaseModel):
    """Inventory recommendation: explicit stats or series-derived forecast."""

    series_id: str | None = None
    horizon_days: int = Field(default=28, ge=1, le=MAX_FORECAST_HORIZON)
    daily_demand_mean: float | None = Field(default=None, ge=0)
    daily_demand_std: float | None = Field(default=None, ge=0)
    inputs: dict = Field(
        default_factory=dict,
        description="InventoryInputs fields (service_level, lead_time_days, ...)",
    )
    persist: bool = True

    @model_validator(mode="after")
    def _require_demand_source(self) -> InventoryRequestBody:
        explicit = (
            self.daily_demand_mean is not None and self.daily_demand_std is not None
        )
        if self.series_id is None and not explicit:
            raise ValueError(
                "Provide either series_id (demand stats derived from the "
                "forecast) or both daily_demand_mean and daily_demand_std."
            )
        return self


class InventoryResponse(BaseModel):
    """Recommendation under stated assumptions, plus provenance."""

    recommendation_id: int | None
    series_id: str | None
    demand_source: Literal["explicit", "series_forecast"]
    recommendation: dict[str, Any]
    warnings: list[str]


class ModelDetail(BaseModel):
    """Full model version record."""

    version: str
    metadata: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    artifacts: list[str]


class DriftWindow(BaseModel):
    """One drift metric block."""

    metric: str
    psi: float
    classification: Literal["stable", "moderate", "significant"]


class DriftResponse(BaseModel):
    """Drift + staleness report with caveats attached."""

    demand: dict[str, Any]
    price: dict[str, Any]
    psi_thresholds: dict[str, Any]
    staleness: dict[str, Any]


class EvaluationReplayResponse(BaseModel):
    """Stored offline validation evaluation (honest replay)."""

    model: dict[str, Any]
    baselines: dict[str, Any]
    comparison_note: str
    replay_note: str


class ScoreRow(BaseModel):
    """One actual-vs-predicted pair."""

    actual_units: float = Field(ge=0)
    predicted_units: float = Field(ge=0)
    horizon: int | None = Field(default=None, ge=1)
    cat_id: str | None = None


class ScoreRequest(BaseModel):
    """Live error-tracking input: rows of actual vs predicted demand."""

    rows: list[ScoreRow] = Field(min_length=1, max_length=200_000)


class ScoreResponse(BaseModel):
    """Scored error report."""

    rows: int
    overall: dict[str, float]
    by_horizon: dict[str, dict[str, float]] | None = None
    by_category: dict[str, dict[str, float]] | None = None
    note: str
