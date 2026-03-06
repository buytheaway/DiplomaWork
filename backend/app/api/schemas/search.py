"""Search request / response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class SearchResult(BaseModel):
    person_id: str
    embedding_id: str
    score: float
    distance: float
    label: str | None = None


class SearchResponse(BaseModel):
    k: int
    model: str
    results: list[SearchResult]
    threshold_used: float = 0.0
    best_score: float | None = None
    best_match_above_threshold: bool = False
    decision: str = "unknown"  # "match" | "unknown"
