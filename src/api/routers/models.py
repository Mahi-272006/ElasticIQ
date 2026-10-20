"""Model registry endpoints: versions, metadata, stored evaluations."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.routers.catalog import get_state
from src.api.schemas import ModelDetail
from src.api.services import AppState

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("")
def list_models(state: Annotated[AppState, Depends(get_state)]) -> dict:
    """All model versions on disk with their recorded metadata."""

    versions = state.list_model_versions()
    return {"total": len(versions), "items": versions}


@router.get("/{version}", response_model=ModelDetail)
def get_model(
    version: str, state: Annotated[AppState, Depends(get_state)]
) -> ModelDetail:
    """One model version with full metadata and evaluation."""

    version_dir = state.models_dir / version
    if not version_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown model version: {version}")

    metadata: dict | None = None
    metadata_path = version_dir / "metadata.json"
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)

    evaluation: dict | None = None
    evaluation_path = version_dir / "evaluation.json"
    if evaluation_path.is_file():
        with evaluation_path.open(encoding="utf-8") as handle:
            evaluation = json.load(handle)

    return ModelDetail(
        version=version,
        metadata=metadata,
        evaluation=evaluation,
        artifacts=sorted(file.name for file in version_dir.iterdir() if file.is_file()),
    )
