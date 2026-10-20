"""SQLAlchemy 2.x ORM models.

Design notes
------------
The raw M5 CSVs/Parquet remain the analytical source of truth; PostgreSQL
(or SQLite in local development) stores *application state*: catalog
references, model versions, forecast runs and outputs, pricing scenarios,
and inventory recommendations. Millions of raw sales values are NOT copied
into the database.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _uuid() -> str:
    return uuid.uuid4().hex


class Store(Base):
    """A physical store in the retail chain (catalog reference)."""

    __tablename__ = "stores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    store_id: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    state_id: Mapped[str] = mapped_column(String(8), nullable=False)

    products: Mapped[list[Product]] = relationship("Product", back_populates="store")


class Product(Base):
    """A product as sold in one store (one forecast series)."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    series_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    item_id: Mapped[str] = mapped_column(String(32), nullable=False)
    dept_id: Mapped[str] = mapped_column(String(32), nullable=False)
    cat_id: Mapped[str] = mapped_column(String(16), nullable=False)
    state_id: Mapped[str] = mapped_column(String(8), nullable=False)
    store_pk: Mapped[int | None] = mapped_column(
        ForeignKey("stores.id", ondelete="SET NULL"), nullable=True
    )

    store: Mapped[Store | None] = relationship("Store", back_populates="products")

    __table_args__ = (
        Index("ix_products_item_id", "item_id"),
        Index("ix_products_cat_id", "cat_id"),
    )


class ModelVersion(Base):
    """A trained model version and its recorded evaluation."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ForecastRun(Base):
    """One persisted forecast request (series + horizon + model)."""

    __tablename__ = "forecast_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    series_id: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    origin_day: Mapped[int] = mapped_column(Integer, nullable=False)
    origin_date: Mapped[str] = mapped_column(String(10), nullable=False)
    assumptions_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        default=_utcnow,
    )

    points: Mapped[list[ForecastPoint]] = relationship(
        "ForecastPoint", back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_forecast_runs_series_id", "series_id"),
        Index("ix_forecast_runs_created_at", "created_at"),
    )


class ForecastPoint(Base):
    """A single predicted day belonging to a forecast run."""

    __tablename__ = "forecast_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_runs.id", ondelete="CASCADE"), nullable=False
    )
    day_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    d: Mapped[str] = mapped_column(String(12), nullable=False)
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    predicted_units: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped[ForecastRun] = relationship("ForecastRun", back_populates="points")

    __table_args__ = (Index("ix_forecast_points_run_id", "run_id"),)


class PricingScenario(Base):
    """A persisted pricing what-if request and its computed outcome."""

    __tablename__ = "pricing_scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    series_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_pricing_scenarios_created_at", "created_at"),)


class InventoryRecommendation(Base):
    """A persisted inventory recommendation under stated assumptions."""

    __tablename__ = "inventory_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    series_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    lead_time_days: Mapped[int] = mapped_column(Integer, nullable=False)
    service_level: Mapped[float] = mapped_column(Float, nullable=False)
    assumed_on_hand: Mapped[float] = mapped_column(Float, nullable=False)
    results_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_inventory_recs_series_id", "series_id"),
        Index("ix_inventory_recs_created_at", "created_at"),
    )
