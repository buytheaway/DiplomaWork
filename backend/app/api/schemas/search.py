"""Search request / response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectedFaceInfo(BaseModel):
    face_index: int = 0
    detection_score: float | None = None
    face_bbox: list[float] | None = None


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
    available_pipelines: list[str] = Field(default_factory=list)
    detected_faces: list[DetectedFaceInfo] = Field(default_factory=list)


class CompareSearchItem(BaseModel):
    pipeline: str
    model: str
    results: list[SearchResult] = Field(default_factory=list)
    faces_detected: int = 0
    matched_faces: int = 0
    threshold_used: float = 0.0
    best_score: float | None = None
    best_match_above_threshold: bool = False
    decision: str = "unknown"
    latency_ms: float | None = None
    error: str | None = None
    detected_faces: list[DetectedFaceInfo] = Field(default_factory=list)


class CompareSearchResponse(BaseModel):
    k: int
    comparisons: list[CompareSearchItem] = Field(default_factory=list)
    available_pipelines: list[str] = Field(default_factory=list)
    fastest_pipeline: str | None = None
