"""Database engine and session management.

The default development database is SQLite (file under ``data/``); set
``ELASTICIQ_DATABASE_URL`` to a PostgreSQL URL for production use (install
``psycopg[binary]`` first). Nothing silently pretends to be connected: engine
errors surface immediately to the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ElasticIQ ORM models."""


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first use."""

    global _engine, _session_factory

    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}

        _engine = create_engine(url, connect_args=connect_args, future=True)
        _session_factory = sessionmaker(
            bind=_engine, autoflush=False, expire_on_commit=False, future=True
        )
        scheme = url.split(":", 1)[0]
        logger.info("Database engine created (scheme=%s)", scheme)

    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the session factory bound to the process-wide engine."""

    get_engine()
    assert _session_factory is not None  # noqa: S101 - set by get_engine
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session that commits or rolls back."""

    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""

    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create tables (development convenience; Alembic is canonical)."""

    from src.db import models  # noqa: F401 - register tables on Base.metadata

    Base.metadata.create_all(get_engine())
    logger.info("Database tables ensured via create_all")


def reset_engine() -> None:
    """Dispose and forget the engine (used by tests to swap databases)."""

    global _engine, _session_factory

    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
