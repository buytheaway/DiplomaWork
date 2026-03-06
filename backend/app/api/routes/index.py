"""Vector index endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_index_manager
from app.api.schemas.index import IndexStatsResponse, RebuildIndexRequest
from app.core.config import settings
from app.services.index.index_manager import IndexManager

router = APIRouter()


@router.get("/index/stats", response_model=IndexStatsResponse)
def index_stats(index_manager: IndexManager = Depends(get_index_manager)) -> IndexStatsResponse:
    data = index_manager.stats()
    data["embedding_backend"] = settings.embedding_backend
    return IndexStatsResponse(**data)


@router.post("/index/rebuild", response_model=IndexStatsResponse)
def rebuild_index(
    payload: RebuildIndexRequest,
    db: Session = Depends(get_db),
    index_manager: IndexManager = Depends(get_index_manager),
) -> IndexStatsResponse:
    stats = index_manager.rebuild(db, payload.index_type, payload.params)
    db.commit()
    return IndexStatsResponse(**stats)
