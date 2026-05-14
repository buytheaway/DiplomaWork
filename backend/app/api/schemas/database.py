"""Database aggregate stats schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingPipelineModelCount(BaseModel):
    pipeline: str
    model_name: str
    count: int


class DatabaseStatsResponse(BaseModel):
    active_persons: int = Field(ge=0)
    active_embeddings: int = Field(ge=0)
    embeddings_by_pipeline_model: list[EmbeddingPipelineModelCount]
