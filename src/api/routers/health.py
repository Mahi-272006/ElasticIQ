"""Health/readiness endpoint.

Reports real resource availability; never fakes a green status when the
database, processed data, or model bundle is missing.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

from src.api.schemas import HealthResponse
from src.db.session import get_engine

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness + readiness: database, processed data, model bundle."""

    app_state = request.app.state.elasticiq

    db_report: dict = {"connected": False}
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        db_report["connected"] = True
    except Exception as exc:  # noqa: BLE001 - health must never raise
        db_report["error"] = f"{type(exc).__name__}: {exc}"

    data_report = app_state.readiness()

    model_report: dict = {
        "version": data_report.get("model_version"),
        "ready": data_report.get("model_ready", False),
    }
    if "model_error" in data_report:
        model_report["error"] = data_report["model_error"]

    all_ready = (
        db_report["connected"]
        and data_report.get("data_ready", False)
        and model_report["ready"]
    )

    return HealthResponse(
        status="ok" if all_ready else "degraded",
        database=db_report,
        data=data_report,
        model=model_report,
    )
