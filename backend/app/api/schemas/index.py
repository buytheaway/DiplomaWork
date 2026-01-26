from typing import Any, Literal

from pydantic import BaseModel, Field


class IndexStatsResponse(BaseModel):
    index_type: str
    params: dict[str, Any]
    embeddings_count: int
    memory_estimate: int
    memory_estimate_bytes: int
    loaded: bool
    is_trained: bool
    file_path: str
    last_snapshot_id: str | None


class RebuildIndexRequest(BaseModel):
    index_type: Literal["flat", "hnsw", "ivfpq"]
    params: dict[str, Any] = Field(default_factory=dict)
