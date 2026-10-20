"""Seed the database from processed M5 dataset artifacts.

Everything inserted here is derived from the real processed dataset
(``products.parquet``) and real training artifacts (``metadata.json`` /
``evaluation.json``). No synthetic "demo" rows are fabricated.

Usage
-----
    python -m src.db.seed                 # stores + products + model versions
    python -m src.db.seed --products-limit 500
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import get_settings
from src.data.pipeline import load_processed
from src.db.models import ModelVersion, Product, Store
from src.db.session import init_db, session_scope

logger = logging.getLogger(__name__)


def seed_stores_and_products(
    session: Session, products: pd.DataFrame, products_limit: int | None = None
) -> dict[str, int]:
    """Insert catalog rows (idempotent: existing ids are skipped)."""

    existing_stores = {
        store_id for (store_id,) in session.execute(select(Store.store_id)).all()
    }
    existing_products = {
        series_id for (series_id,) in session.execute(select(Product.series_id)).all()
    }

    stores_added = 0
    store_pk: dict[str, int] = {}

    for store_id, state_id in sorted(
        products[["store_id", "state_id"]].drop_duplicates().itertuples(index=False)
    ):
        if store_id in existing_stores:
            continue
        store = Store(store_id=store_id, state_id=state_id)
        session.add(store)
        session.flush()
        store_pk[store_id] = store.id
        stores_added += 1

    # Ensure pks for pre-existing stores are resolvable.
    for existing_id, pk in session.execute(select(Store.store_id, Store.id)).all():
        store_pk.setdefault(existing_id, pk)

    frame = products
    if products_limit is not None:
        frame = frame.head(products_limit)

    products_added = 0
    batch: list[Product] = []
    for row in frame.to_dict("records"):
        if row["id"] in existing_products:
            continue
        batch.append(
            Product(
                series_id=row["id"],
                item_id=row["item_id"],
                dept_id=row["dept_id"],
                cat_id=row["cat_id"],
                state_id=row["state_id"],
                store_pk=store_pk.get(row["store_id"]),
            )
        )
        products_added += 1

    session.add_all(batch)
    session.flush()

    return {"stores_added": stores_added, "products_added": products_added}


def seed_model_versions(session: Session, models_dir: Path) -> list[str]:
    """Insert a ModelVersion row per trained version on disk (idempotent)."""

    if not models_dir.is_dir():
        logger.warning("Models directory %s does not exist; skipping", models_dir)
        return []

    existing = {
        version for (version,) in session.execute(select(ModelVersion.version)).all()
    }
    added: list[str] = []

    for path in sorted(models_dir.iterdir()):
        if not path.is_dir() or path.name in existing:
            continue

        metadata_path = path / "metadata.json"
        evaluation_path = path / "evaluation.json"
        if not metadata_path.is_file():
            logger.warning("Version %s has no metadata.json; skipping", path.name)
            continue

        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        evaluation: dict = {}
        if evaluation_path.is_file():
            with evaluation_path.open(encoding="utf-8") as handle:
                evaluation = json.load(handle)

        trained_at = metadata.get("trained_at")
        session.add(
            ModelVersion(
                version=path.name,
                algorithm=str(metadata.get("algorithm", "unknown")),
                metrics_json=json.dumps(evaluation),
                metadata_json=json.dumps(metadata),
                trained_at=pd.Timestamp(trained_at).to_pydatetime()
                if trained_at
                else None,
            )
        )
        added.append(path.name)

    session.flush()
    return added


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--products-limit",
        type=int,
        default=None,
        help="Insert only the first N products (default: all catalog rows)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    settings = get_settings()
    init_db()

    products = load_processed("products", settings.processed_data_dir)

    with session_scope() as session:
        counts = seed_stores_and_products(session, products, args.products_limit)
        versions = seed_model_versions(session, settings.models_dir)

    logger.info(
        "Seed complete: %s stores added, %s products added, model versions: %s",
        counts["stores_added"],
        counts["products_added"],
        versions or "none",
    )


if __name__ == "__main__":
    main()
