"""ElasticIQ FastAPI application factory.

Design
------
- One shared :class:`~src.api.services AppState` per process: processed
  Parquet datasets and the trained model bundle are loaded lazily on first
  access and cached — never re-read per request.
- Database schema is ensured at startup for development convenience;
  Alembic (``alembic upgrade head``) remains the canonical migration path,
  especially for PostgreSQL.
- Startup never silently swallows failures: missing data or a missing model
  leaves the API running in a degraded state that /health reports honestly,
  while endpoints that need the resource return 404/503 with real messages.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import (
    catalog,
    forecasts,
    health,
    inventory,
    models,
    monitoring,
    overview,
    pricing,
    trends,
)
from src.api.services import AppState
from src.config import get_settings
from src.db.session import init_db

logger = logging.getLogger("elasticiq.api")

API_DESCRIPTION = """\
ElasticIQ — demand forecasting, pricing intelligence, and inventory
decision support over the M5 retail dataset.

Honesty contract:
- Forecast metrics are holdout WAPE on chronological validation days,
  reported next to naive baselines evaluated on identical inputs.
- Pricing endpoints require an explicit demand-response assumption; the
  observational M5 data does not identify causal price effects.
- Inventory recommendations require user-supplied stock parameters; the
  dataset contains no warehouse state.
"""


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (idempotent for tests)."""

    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _startup_checks(app: FastAPI) -> None:
    """Ensure dev database tables exist; report readiness at startup."""

    try:
        init_db()
        logger.info("Database schema ensured at startup")
    except Exception:
        logger.exception(
            "Database not reachable at startup; API will serve "
            "read-only endpoints and report degraded health"
        )

    readiness = app.state.elasticiq.readiness()
    logger.info(
        "Startup readiness: data_ready=%s model_ready=%s model=%s",
        readiness.get("data_ready"),
        readiness.get("model_ready"),
        readiness.get("model_version"),
    )


def _mount_frontend(app: FastAPI, settings) -> None:
    """Serve the built React dashboard from frontend/dist when present.

    Registered after the API routers so /api/** and /health always win;
    unknown paths fall back to index.html (client-side routing).
    """

    dist = settings.project_root / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        logger.info("No frontend build at %s; API-only mode", dist)
        return

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        target = dist / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(dist / "index.html")

    logger.info("Serving frontend build from %s", dist)


def create_app(state: AppState | None = None) -> FastAPI:
    """Build the ElasticIQ API application."""

    configure_logging()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        _startup_checks(app)
        yield

    app = FastAPI(
        title="ElasticIQ API",
        version="1.0.0",
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.elasticiq = state or AppState()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Structured one-line request log with duration."""

        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "http_request method=%s path=%s status=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError):
        """Unknown series/keys surface as 404 with the real message."""

        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def file_error_handler(request: Request, exc: FileNotFoundError):
        """Missing processed data or model artifacts surface as 503."""

        return JSONResponse(status_code=503, content={"detail": str(exc)})

    app.include_router(health.router)
    app.include_router(overview.router)
    app.include_router(catalog.router)
    app.include_router(trends.router)
    app.include_router(forecasts.router)
    app.include_router(pricing.router)
    app.include_router(inventory.router)
    app.include_router(models.router)
    app.include_router(monitoring.router)

    @app.get("/", tags=["health"])
    def root() -> dict:
        """Service root with pointers to docs and versioned endpoints."""

        return {
            "service": "elasticiq-api",
            "version": "1.0.0",
            "docs": "/docs",
            "api_base": "/api/v1",
            "health": "/health",
        }

    _mount_frontend(app, settings)

    return app


app = create_app()
