from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_index_manager
from app.api.schemas.index import IndexStatsResponse, RebuildIndexRequest

router = APIRouter()


@router.get("/index/stats", response_model=IndexStatsResponse)
def index_stats(index_manager=Depends(get_index_manager)) -> IndexStatsResponse:
    return IndexStatsResponse(**index_manager.stats())


@router.post("/index/rebuild", response_model=IndexStatsResponse)
def rebuild_index(
    payload: RebuildIndexRequest,
    db: Session = Depends(get_db),
    index_manager=Depends(get_index_manager),
) -> IndexStatsResponse:
    stats = index_manager.rebuild(db, payload.index_type, payload.params)
    db.commit()
    return IndexStatsResponse(**stats)
