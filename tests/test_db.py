"""Persistence-layer tests (SQLAlchemy models, cascade, seed idempotency)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from src.db.models import ForecastPoint, ForecastRun, Product, Store
from src.db.seed import seed_stores_and_products
from src.db.session import (
    get_engine,
    get_settings,
    init_db,
    reset_engine,
    session_scope,
)


@pytest.fixture
def db_env(tmp_path: Path, monkeypatch):
    """Point the engine at a throwaway SQLite file and create all tables."""

    monkeypatch.setenv(
        "ELASTICIQ_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    )
    get_settings.cache_clear()
    reset_engine()
    init_db()
    yield
    reset_engine()
    get_settings.cache_clear()


def test_init_db_creates_all_tables(db_env) -> None:
    inspector = inspect(get_engine())
    tables = set(inspector.get_table_names())
    assert {
        "stores",
        "products",
        "model_versions",
        "forecast_runs",
        "forecast_points",
        "pricing_scenarios",
        "inventory_recommendations",
    } <= tables


def test_store_and_product_roundtrip(db_env) -> None:
    with session_scope() as session:
        store = Store(store_id="CA_1", state_id="CA")
        session.add(store)
        session.flush()
        session.add(
            Product(
                series_id="FOODS_3_090_CA_1",
                item_id="FOODS_3_090",
                dept_id="FOODS_3",
                cat_id="FOODS",
                state_id="CA",
                store_pk=store.id,
            )
        )

    with session_scope() as session:
        product = session.scalar(
            select(Product).where(Product.series_id == "FOODS_3_090_CA_1")
        )
        assert product is not None
        assert product.store is not None
        assert product.store.store_id == "CA_1"


def test_duplicate_series_id_rejected(db_env) -> None:
    with session_scope() as session:
        session.add(
            Product(
                series_id="X_1",
                item_id="X",
                dept_id="X_1",
                cat_id="FOODS",
                state_id="CA",
            )
        )

    with pytest.raises(IntegrityError):
        with session_scope() as session:
            session.add(
                Product(
                    series_id="X_1",
                    item_id="X",
                    dept_id="X_1",
                    cat_id="FOODS",
                    state_id="CA",
                )
            )


def test_forecast_run_cascade_delete(db_env) -> None:
    with session_scope() as session:
        run = ForecastRun(
            series_id="FOODS_3_090_CA_1",
            horizon_days=2,
            model_version="stub_v1",
            origin_day=60,
            origin_date="2011-03-29",
            assumptions_json="[]",
            points=[
                ForecastPoint(
                    day_idx=61, d="d_61", date="2011-03-30", predicted_units=3.0
                ),
                ForecastPoint(
                    day_idx=62, d="d_62", date="2011-03-31", predicted_units=2.0
                ),
            ],
        )
        session.add(run)
        session.flush()
        run_id = run.id

    with session_scope() as session:
        run = session.get(ForecastRun, run_id)
        assert len(run.points) == 2
        assert run.created_at is not None

    with session_scope() as session:
        run = session.get(ForecastRun, run_id)
        session.delete(run)

    with session_scope() as session:
        remaining = session.scalars(
            select(ForecastPoint).where(ForecastPoint.run_id == run_id)
        ).all()
        assert remaining == []


def test_seed_is_idempotent(db_env) -> None:
    products = pd.DataFrame(
        [
            ("FOODS_3_090_CA_1", "FOODS_3_090", "FOODS_3", "FOODS", "CA_1", "CA"),
            ("FOODS_3_090_TX_1", "FOODS_3_090", "FOODS_3", "FOODS", "TX_1", "TX"),
        ],
        columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )

    with session_scope() as session:
        first = seed_stores_and_products(session, products)
    assert first == {"stores_added": 2, "products_added": 2}

    with session_scope() as session:
        second = seed_stores_and_products(session, products)
    assert second == {"stores_added": 0, "products_added": 0}

    with session_scope() as session:
        assert len(session.scalars(select(Product)).all()) == 2
        assert len(session.scalars(select(Store)).all()) == 2


def test_seed_respects_products_limit(db_env) -> None:
    products = pd.DataFrame(
        [
            ("A_CA_1", "A", "A_1", "FOODS", "CA_1", "CA"),
            ("B_CA_1", "B", "B_1", "FOODS", "CA_1", "CA"),
            ("C_CA_1", "C", "C_1", "FOODS", "CA_1", "CA"),
        ],
        columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )

    with session_scope() as session:
        counts = seed_stores_and_products(session, products, products_limit=2)

    assert counts == {"stores_added": 1, "products_added": 2}
