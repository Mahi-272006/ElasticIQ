"""Monitoring endpoints: drift, staleness, evaluation replay, live scoring."""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.api.routers.catalog import get_state
from src.api.schemas import DriftResponse, ScoreRequest, ScoreResponse
from src.api.services import AppState
from src.monitoring.drift import drift_report
from src.monitoring.tracking import (
    evaluation_report,
    score_predictions,
    staleness_report,
)

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@router.get("/drift", response_model=DriftResponse)
def drift(state: Annotated[AppState, Depends(get_state)]) -> DriftResponse:
    """PSI drift between recent and prior windows, plus staleness report."""

    report = drift_report(state.history_tail, state.price_stats, window_days=28)

    try:
        staleness = staleness_report(
            state.models_dir,
            last_data_date=state.last_data_date(),
            model_version=state.latest_model_version(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return DriftResponse(**report, staleness=staleness)


@router.get("/evaluation/{version}")
def evaluation(version: str, state: Annotated[AppState, Depends(get_state)]) -> dict:
    """Stored offline validation evaluation for one model version."""

    try:
        return evaluation_report(state.models_dir, version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/score", response_model=ScoreResponse)
def score(body: ScoreRequest) -> ScoreResponse:
    """Score actual vs predicted demand (live error-tracking entry point)."""

    frame = pd.DataFrame([row.model_dump() for row in body.rows])
    frame = frame.dropna(axis=1, how="all")

    if "horizon" in frame.columns and frame["horizon"].isna().all():
        frame = frame.drop(columns=["horizon"])
    if "cat_id" in frame.columns and frame["cat_id"].isna().all():
        frame = frame.drop(columns=["cat_id"])

    try:
        report = score_predictions(frame)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ScoreResponse(**report)
