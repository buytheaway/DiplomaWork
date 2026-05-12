"""Face search endpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_pipeline_registry
from app.api.schemas.search import (
    DetectedFaceInfo,
    LatencyBreakdown,
    SearchResponse,
    SearchResult,
)
from app.core.config import settings
from app.services.embeddings.interface import (
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.services.index.index_manager import IndexMatch
from app.services.runtime.pipeline_registry import PipelineRegistry
from app.services.storage.audit import record_audit_event
from app.services.storage.repositories import EmbeddingRepo
from app.services.storage.uploads import (
    UploadValidationError,
    allowed_content_types,
    read_image_upload,
)

router = APIRouter()


@dataclass(frozen=True)
class RawFaceOutcome:
    face_index: int
    detection_score: float | None
    face_bbox: tuple[float, float, float, float] | None
    matches: list[IndexMatch]
    best_score: float | None
    best_match_above_threshold: bool
    decision: str


@dataclass(frozen=True)
class RawSearchOutcome:
    pipeline: str
    model: str
    faces: list[RawFaceOutcome]
    latency_ms: float
    threshold_used: float
    best_score: float | None
    best_match_above_threshold: bool
    decision: str
    error: str | None = None
    latency_breakdown: LatencyBreakdown | None = None


def _map_face_error(exc: Exception) -> HTTPException:
    if isinstance(exc, UploadValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, InvalidImageError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (NoFaceDetectedError, MultipleFacesDetectedError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="Embedding extraction failed")


def _run_pipeline_search(runtime, image_bytes: bytes, k: int) -> RawSearchOutcome:
    start = time.perf_counter()

    # --- Stage 1: detect + embed ---
    t_detect = time.perf_counter()
    detected_faces = runtime.extractor.extract_embeddings(image_bytes)
    t_embed_done = time.perf_counter()
    detect_embed_ms = (t_embed_done - t_detect) * 1000

    threshold = settings.match_threshold
    faces: list[RawFaceOutcome] = []
    index_count = runtime.index_manager.count()
    effective_k = min(k, index_count) if index_count > 0 else 0

    # --- Stage 2: search ---
    t_search = time.perf_counter()
    for face_index, detected in enumerate(detected_faces):
        matches: list[IndexMatch] = []
        if effective_k > 0:
            matches = runtime.index_manager.search(detected.embedding, effective_k)

        best_score = matches[0].score if matches else None
        above = best_score is not None and best_score >= threshold
        faces.append(
            RawFaceOutcome(
                face_index=face_index,
                detection_score=detected.detection_score,
                face_bbox=detected.bbox,
                matches=matches,
                best_score=best_score,
                best_match_above_threshold=above,
                decision="match" if above else "unknown",
            )
        )
    t_search_done = time.perf_counter()
    search_ms = (t_search_done - t_search) * 1000

    best_score = max(
        (face.best_score for face in faces if face.best_score is not None),
        default=None,
    )
    any_match = any(face.best_match_above_threshold for face in faces)
    latency_ms = (time.perf_counter() - start) * 1000

    latency_breakdown = LatencyBreakdown(
        detect_ms=round(detect_embed_ms, 2),
        embed_ms=None,  # detect+embed are currently fused
        search_ms=round(search_ms, 2),
        total_ms=round(latency_ms, 2),
    )

    return RawSearchOutcome(
        pipeline=runtime.key,
        model=runtime.extractor.model_name,
        faces=faces,
        latency_ms=latency_ms,
        threshold_used=threshold,
        best_score=best_score,
        best_match_above_threshold=any_match,
        decision="match" if any_match else "unknown",
        latency_breakdown=latency_breakdown,
    )


def _hydrate_results(
    db: Session,
    pipeline: str,
    faces: list[RawFaceOutcome],
    threshold: float,
) -> list[SearchResult]:
    repo = EmbeddingRepo(db)
    embedding_ids = [
        match.embedding_id
        for face in faces
        for match in face.matches
        if match.score >= threshold
    ]
    if not embedding_ids:
        return []

    unique_embedding_ids: list[str] = []
    seen_embedding_ids: set[str] = set()
    for embedding_id in embedding_ids:
        if embedding_id in seen_embedding_ids:
            continue
        seen_embedding_ids.add(embedding_id)
        unique_embedding_ids.append(embedding_id)

    rows = repo.get_embeddings_with_person(unique_embedding_ids, pipeline=pipeline)
    lookup = {str(row.id): row for row in rows}

    results: list[SearchResult] = []
    for face in faces:
        bbox = list(face.face_bbox) if face.face_bbox is not None else None
        for match in face.matches:
            if match.score < threshold:
                continue
            row = lookup.get(match.embedding_id)
            if row is None or row.person is None:
                continue
            # Skip deactivated embeddings or soft-deleted persons
            if not row.is_active or row.person.status != "active" or row.pipeline != pipeline:
                continue
            results.append(
                SearchResult(
                    pipeline=pipeline,
                    face_index=face.face_index,
                    detection_score=face.detection_score,
                    face_bbox=bbox,
                    person_id=str(row.person.id),
                    embedding_id=match.embedding_id,
                    score=match.score,
                    distance=match.distance,
                    label=row.person.label,
                )
            )
    return results


def _response_from_outcome(
    db: Session,
    outcome: RawSearchOutcome,
    k: int,
    available_pipelines: list[str],
) -> SearchResponse:
    results = _hydrate_results(
        db,
        outcome.pipeline,
        outcome.faces,
        outcome.threshold_used,
    )
    matched_faces = sum(1 for face in outcome.faces if face.best_match_above_threshold)
    detected_faces = [
        DetectedFaceInfo(
            face_index=face.face_index,
            detection_score=face.detection_score,
            face_bbox=list(face.face_bbox) if face.face_bbox is not None else None,
        )
        for face in outcome.faces
    ]
    return SearchResponse(
        k=k,
        model=outcome.model,
        results=results,
        faces_detected=len(outcome.faces),
        matched_faces=matched_faces,
        threshold_used=outcome.threshold_used,
        best_score=outcome.best_score,
        best_match_above_threshold=outcome.best_match_above_threshold,
        decision=outcome.decision,
        pipeline=outcome.pipeline,
        latency_ms=outcome.latency_ms,
        available_pipelines=available_pipelines,
        detected_faces=detected_faces,
        latency_breakdown=outcome.latency_breakdown,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: Request,
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=100),
    pipeline: Literal["pretrained", "custom"] | None = Query(None),
    db: Session = Depends(get_db),
    registry: PipelineRegistry = Depends(get_pipeline_registry),
) -> SearchResponse:
    try:
        image_bytes = await read_image_upload(
            file,
            max_bytes=settings.max_upload_bytes,
            allowed_types=allowed_content_types(settings.allowed_image_content_types),
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        runtime = registry.resolve_search(pipeline)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        outcome = _run_pipeline_search(runtime, image_bytes, k)
    except Exception as exc:  # noqa: BLE001
        raise _map_face_error(exc) from exc

    response = _response_from_outcome(db, outcome, k, registry.available_pipelines())
    if settings.enable_search_audit:
        record_audit_event(
            db,
            request,
            event_type="search",
            status_code=200,
            details={
                "pipeline": outcome.pipeline,
                "faces_detected": response.faces_detected,
                "matched_faces": response.matched_faces,
                "decision": response.decision,
                "k": k,
            },
        )
    return response
