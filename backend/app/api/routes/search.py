"""Face search endpoints."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_pipeline_registry
from app.api.schemas.search import (
    CompareSearchItem,
    CompareSearchResponse,
    DetectedFaceInfo,
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
from app.services.storage.repositories import EmbeddingRepo

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


def _map_face_error(exc: Exception) -> HTTPException:
    if isinstance(exc, InvalidImageError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (NoFaceDetectedError, MultipleFacesDetectedError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="Embedding extraction failed")


def _run_pipeline_search(runtime, image_bytes: bytes, k: int) -> RawSearchOutcome:
    start = time.perf_counter()
    detected_faces = runtime.extractor.extract_embeddings(image_bytes)
    threshold = settings.match_threshold
    faces: list[RawFaceOutcome] = []

    for face_index, detected in enumerate(detected_faces):
        matches: list[IndexMatch] = []
        if runtime.index_manager.count() > 0:
            matches = runtime.index_manager.search(detected.embedding, k)

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

    best_score = max(
        (face.best_score for face in faces if face.best_score is not None),
        default=None,
    )
    any_match = any(face.best_match_above_threshold for face in faces)
    latency_ms = (time.perf_counter() - start) * 1000

    return RawSearchOutcome(
        pipeline=runtime.key,
        model=runtime.extractor.model_name,
        faces=faces,
        latency_ms=latency_ms,
        threshold_used=threshold,
        best_score=best_score,
        best_match_above_threshold=any_match,
        decision="match" if any_match else "unknown",
    )


def _hydrate_results(
    db: Session,
    pipeline: str,
    faces: list[RawFaceOutcome],
) -> list[SearchResult]:
    repo = EmbeddingRepo(db)
    embedding_ids = [
        match.embedding_id
        for face in faces
        for match in face.matches
    ]
    rows = repo.get_embeddings_with_person(embedding_ids)
    lookup = {str(row.id): row for row in rows}

    results: list[SearchResult] = []
    for face in faces:
        bbox = list(face.face_bbox) if face.face_bbox is not None else None
        for match in face.matches:
            row = lookup.get(match.embedding_id)
            if row is None or row.person is None:
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
    results = _hydrate_results(db, outcome.pipeline, outcome.faces)
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
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=100),
    pipeline: Literal["pretrained", "custom"] | None = Query(None),
    db: Session = Depends(get_db),
    registry: PipelineRegistry = Depends(get_pipeline_registry),
) -> SearchResponse:
    image_bytes = await file.read()
    try:
        runtime = registry.resolve_search(pipeline)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        outcome = _run_pipeline_search(runtime, image_bytes, k)
    except Exception as exc:  # noqa: BLE001
        raise _map_face_error(exc) from exc

    return _response_from_outcome(db, outcome, k, registry.available_pipelines())


@router.post("/search/compare", response_model=CompareSearchResponse)
async def search_compare(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=100),
    db: Session = Depends(get_db),
    registry: PipelineRegistry = Depends(get_pipeline_registry),
) -> CompareSearchResponse:
    image_bytes = await file.read()
    available = registry.available_pipelines()
    if "pretrained" not in available or "custom" not in available:
        raise HTTPException(
            status_code=400,
            detail="Compare mode requires both 'pretrained' and 'custom' pipelines",
        )

    runtimes = [registry.get("pretrained"), registry.get("custom")]
    outcomes: list[RawSearchOutcome] = []

    def _capture(runtime) -> RawSearchOutcome:
        started = time.perf_counter()
        try:
            return _run_pipeline_search(runtime, image_bytes, k)
        except Exception as exc:  # noqa: BLE001
            mapped = _map_face_error(exc)
            return RawSearchOutcome(
                pipeline=runtime.key,
                model=runtime.extractor.model_name,
                faces=[],
                latency_ms=(time.perf_counter() - started) * 1000,
                threshold_used=settings.match_threshold,
                best_score=None,
                best_match_above_threshold=False,
                decision="error",
                error=mapped.detail,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_capture, runtime) for runtime in runtimes]
        for future in futures:
            outcomes.append(future.result())

    comparisons: list[CompareSearchItem] = []
    successful: list[RawSearchOutcome] = []
    for outcome in outcomes:
        if outcome.error is None:
            successful.append(outcome)
        comparisons.append(
            CompareSearchItem(
                pipeline=outcome.pipeline,
                model=outcome.model,
                results=_hydrate_results(db, outcome.pipeline, outcome.faces),
                faces_detected=len(outcome.faces),
                matched_faces=sum(
                    1 for face in outcome.faces if face.best_match_above_threshold
                ),
                threshold_used=outcome.threshold_used,
                best_score=outcome.best_score,
                best_match_above_threshold=outcome.best_match_above_threshold,
                decision=outcome.decision,
                latency_ms=outcome.latency_ms,
                error=outcome.error,
                detected_faces=[
                    DetectedFaceInfo(
                        face_index=face.face_index,
                        detection_score=face.detection_score,
                        face_bbox=list(face.face_bbox) if face.face_bbox is not None else None,
                    )
                    for face in outcome.faces
                ],
            )
        )

    fastest = min(successful, key=lambda item: item.latency_ms).pipeline if successful else None
    return CompareSearchResponse(
        k=k,
        comparisons=comparisons,
        available_pipelines=available,
        fastest_pipeline=fastest,
    )
