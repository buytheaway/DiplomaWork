"""Vector index endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_pipeline_registry
from app.api.schemas.index import IndexStatsResponse, RebuildIndexRequest
from app.services.runtime.pipeline_registry import PipelineRegistry

router = APIRouter()


@router.get("/index/stats", response_model=IndexStatsResponse)
def index_stats(
    pipeline: Literal["pretrained", "custom"] | None = Query(None),
    registry: PipelineRegistry = Depends(get_pipeline_registry),
) -> IndexStatsResponse:
    try:
        runtime = registry.resolve_search(pipeline)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data = runtime.index_manager.stats()
    data["embedding_backend"] = runtime.backend
    data["pipeline"] = runtime.key
    data["available_pipelines"] = registry.available_pipelines()
    return IndexStatsResponse(**data)


@router.post("/index/rebuild", response_model=IndexStatsResponse)
def rebuild_index(
    payload: RebuildIndexRequest,
    db: Session = Depends(get_db),
    registry: PipelineRegistry = Depends(get_pipeline_registry),
) -> IndexStatsResponse:
    try:
        runtime = registry.resolve_search(payload.pipeline)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stats = runtime.index_manager.rebuild(db, payload.index_type, payload.params)
    db.commit()
    stats["embedding_backend"] = runtime.backend
    stats["pipeline"] = runtime.key
    stats["available_pipelines"] = registry.available_pipelines()
    return IndexStatsResponse(**stats)
