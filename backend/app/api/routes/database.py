"""Database aggregate statistics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.schemas.database import (
    DatabaseStatsResponse,
    EmbeddingPipelineModelCount,
)
from app.services.storage.repositories import EmbeddingRepo, PersonRepo

router = APIRouter()


@router.get("/database/stats", response_model=DatabaseStatsResponse)
def database_stats(db: Session = Depends(get_db)) -> DatabaseStatsResponse:
    person_repo = PersonRepo(db)
    embedding_repo = EmbeddingRepo(db)
    grouped = [
        EmbeddingPipelineModelCount(
            pipeline=pipeline,
            model_name=model,
            count=count,
        )
        for pipeline, model, count in embedding_repo.count_active_embeddings_by_pipeline_model()
    ]
    return DatabaseStatsResponse(
        active_persons=person_repo.count_active_persons(),
        active_embeddings=embedding_repo.count_active_embeddings(),
        embeddings_by_pipeline_model=grouped,
    )
