"""Vector index endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_pipeline_registry, require_admin
from app.api.schemas.index import IndexStatsResponse, RebuildIndexRequest
from app.services.index.index_manager import IndexRebuildTooLargeError
from app.services.runtime.pipeline_registry import PipelineRegistry
from app.services.storage.audit import record_audit_event

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
    request: Request,
    payload: RebuildIndexRequest,
    db: Session = Depends(get_db),
    registry: PipelineRegistry = Depends(get_pipeline_registry),
    _admin: str = Depends(require_admin),
) -> IndexStatsResponse:
    try:
        runtime = registry.resolve_search(payload.pipeline)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        stats = runtime.index_manager.rebuild(db, payload.index_type, payload.params)
    except IndexRebuildTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    db.commit()
    stats["embedding_backend"] = runtime.backend
    stats["pipeline"] = runtime.key
    stats["available_pipelines"] = registry.available_pipelines()
    record_audit_event(
        db,
        request,
        event_type="rebuild_index",
        status_code=200,
        details={
            "pipeline": runtime.key,
            "index_type": payload.index_type,
            "embeddings_count": stats["embeddings_count"],
        },
    )
    return IndexStatsResponse(**stats)
