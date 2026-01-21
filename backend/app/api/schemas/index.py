from typing import Any, Literal

from pydantic import BaseModel, Field


class IndexStatsResponse(BaseModel):
    index_type: str
    params: dict[str, Any]
    embeddings_count: int
    memory_estimate: int
    loaded: bool


class RebuildIndexRequest(BaseModel):
    index_type: Literal["flat", "hnsw", "ivfpq"]
    params: dict[str, Any] = Field(default_factory=dict)
