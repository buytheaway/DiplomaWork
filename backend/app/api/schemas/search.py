"""Search request / response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectedFaceInfo(BaseModel):
    face_index: int = 0
    detection_score: float | None = None
    face_bbox: list[float] | None = None


class LatencyBreakdown(BaseModel):
    upload_ms: float | None = None
    decode_ms: float | None = None
    resize_ms: float | None = None
    detect_ms: float | None = None
    embed_ms: float | None = None
    align_ms: float | None = None
    faiss_ms: float | None = None
    search_ms: float | None = None
    rerank_ms: float | None = None
    db_ms: float | None = None
    response_ms: float | None = None
    total_ms: float | None = None


class SearchResult(BaseModel):
    pipeline: str | None = None
    face_index: int = 0
    detection_score: float | None = None
    face_bbox: list[float] | None = None
    person_id: str
    embedding_id: str
    score: float
    distance: float
    label: str | None = None
    support_count: int = 1


class SearchResponse(BaseModel):
    k: int
    model: str
    results: list[SearchResult]
    faces_detected: int = 0
    matched_faces: int = 0
    threshold_used: float = 0.0
    best_score: float | None = None
    best_match_above_threshold: bool = False
    decision: str = "unknown"  # "match" | "unknown"
    pipeline: str | None = None
    latency_ms: float | None = None
    search_mode: str | None = None
    candidate_k: int | None = None
    fallback_reason: str | None = None
    multi_face_enabled: bool = True
    faces_processed: int = 0
    available_pipelines: list[str] = Field(default_factory=list)
    detected_faces: list[DetectedFaceInfo] = Field(default_factory=list)
    latency_breakdown: LatencyBreakdown | None = None
