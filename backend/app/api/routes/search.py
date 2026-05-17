"""Face search endpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
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
from app.security.crypto import decrypt_embedding_payload
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
    query_vector: np.ndarray
    matches: list[IndexMatch]
    best_score: float | None
    best_match_above_threshold: bool
    decision: str
    search_mode: str
    candidate_k: int
    fallback_reason: str | None


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
    search_mode: str
    candidate_k: int
    fallback_reason: str | None
    multi_face_enabled: bool
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


def _threshold_for_pipeline(pipeline: str, *, is_live: bool = False) -> float:
    if (
        pipeline == "custom"
        and is_live
        and settings.custom_live_match_threshold is not None
    ):
        return settings.custom_live_match_threshold
    if pipeline == "custom" and settings.custom_match_threshold is not None:
        return settings.custom_match_threshold
    if pipeline == "pretrained" and settings.pretrained_match_threshold is not None:
        return settings.pretrained_match_threshold
    return settings.match_threshold


def _best_score(matches: list[IndexMatch]) -> float | None:
    return matches[0].score if matches else None


def _fallback_reason(best_score: float | None, threshold: float) -> str | None:
    if not settings.search_dynamic_enabled:
        return None
    if best_score is None:
        return "no_candidates"
    if best_score < threshold:
        return "below_threshold"
    if best_score < threshold + settings.search_fallback_margin:
        return "near_threshold"
    return None


def _search_index_once(runtime, vector, candidate_k: int, nprobe: int) -> list[IndexMatch]:
    index_count = runtime.index_manager.count()
    effective_k = min(max(1, candidate_k), index_count) if index_count > 0 else 0
    if effective_k <= 0:
        return []
    return runtime.index_manager.search(
        vector,
        effective_k,
        search_params={"nprobe": nprobe},
    )


def _search_index_dynamic(
    runtime,
    vector,
    visible_k: int,
    threshold: float,
    *,
    is_live: bool,
) -> tuple[list[IndexMatch], str, int, str | None]:
    if is_live:
        live_k = max(visible_k, settings.search_candidate_k_fast)
        matches = _search_index_once(
            runtime,
            vector,
            live_k,
            settings.ivfpq_nprobe_fast,
        )
        return matches, "live_fast", live_k, None

    if not settings.search_dynamic_enabled:
        candidate_k = max(visible_k, settings.search_candidate_k)
        matches = _search_index_once(runtime, vector, candidate_k, settings.ivfpq_nprobe)
        return matches, "single", candidate_k, None

    fast_k = max(visible_k, settings.search_candidate_k_fast)
    fast_matches = _search_index_once(
        runtime,
        vector,
        fast_k,
        settings.ivfpq_nprobe_fast,
    )
    fast_best = _best_score(fast_matches)

    fallback_reason = _fallback_reason(fast_best, threshold)
    if fallback_reason is None:
        return fast_matches, "fast", fast_k, None

    safe_k = max(fast_k, visible_k, settings.search_candidate_k_safe)
    safe_matches = _search_index_once(
        runtime,
        vector,
        safe_k,
        settings.ivfpq_nprobe_safe,
    )
    safe_best = _best_score(safe_matches)
    if safe_best is not None and (fast_best is None or safe_best >= fast_best):
        return safe_matches, "safe_fallback", safe_k, fallback_reason
    return fast_matches, "fast", fast_k, None


def _is_supported_person_match(
    pipeline: str,
    score: float,
    support_count: int,
    threshold: float,
    label: str | None = None,
    *,
    multi_face: bool = False,
) -> bool:
    if _is_scale_label(label):
        return False
    if score < threshold:
        return False

    if pipeline != "custom":
        if multi_face:
            if score >= 0.56:
                return True
            if score >= 0.50 and support_count >= 2:
                return True
            return score >= 0.46 and support_count >= 3

        # On a very large demo index, a single weak nearest vector is not
        # enough evidence. Prefer no-match over assigning a random identity.
        if score >= 0.54:
            return True
        if score >= 0.46 and support_count >= 2:
            return True
        return score >= 0.40 and support_count >= 3

    # The custom pipeline is noisier than the pretrained extractor on webcam
    # crops, so it also needs either stronger score or repeated support.
    if multi_face:
        if score >= 0.38:
            return True
        return score >= 0.34 and support_count >= 2

    if score >= 0.36:
        return True
    if score >= 0.32 and support_count >= 2:
        return True
    return score >= 0.30 and support_count >= 4


def _is_scale_label(label: str | None) -> bool:
    if not label:
        return False
    normalized = label.upper()
    return (
        label.startswith("Dataset Identity ")
        or label.startswith("Scale Person ")
        or "#SCALE" in normalized
    )


def _is_ambiguous_person_ranking(
    pipeline: str,
    best: SearchResult,
    second: SearchResult | None,
    *,
    multi_face: bool = False,
) -> bool:
    if second is None:
        return False

    margin = best.score - second.score
    if pipeline == "custom":
        if multi_face:
            if best.score < 0.38 and margin < 0.08:
                return True
            return best.support_count <= second.support_count and margin < 0.06
        if best.score < 0.34 and margin < 0.05:
            return True
        return best.support_count <= second.support_count and margin < 0.035

    if multi_face:
        return best.score < 0.56 and margin < 0.07

    return best.score < 0.50 and margin < 0.055


def _identity_key(result: SearchResult) -> str:
    label = (result.label or "").strip()
    if label:
        return f"label:{label.casefold()}"
    return f"person:{result.person_id}"


def _deduplicate_identity_across_faces(results: list[SearchResult]) -> list[SearchResult]:
    best_by_identity: dict[str, SearchResult] = {}
    for result in results:
        key = _identity_key(result)
        current = best_by_identity.get(key)
        if current is None:
            best_by_identity[key] = result
            continue
        current_rank = (current.score, current.support_count)
        result_rank = (result.score, result.support_count)
        if result_rank > current_rank:
            best_by_identity[key] = result

    return sorted(
        best_by_identity.values(),
        key=lambda item: (item.face_index, -item.score, -item.support_count),
    )


def _decode_embedding_vector(row, expected_dim: int) -> np.ndarray | None:
    try:
        raw = decrypt_embedding_payload(row.vector)
        vector = np.frombuffer(raw, dtype=np.float32)
    except Exception:
        return None
    if vector.shape[0] != expected_dim:
        return None
    vector = vector.astype(np.float32, copy=True)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return None
    return vector / norm


def _exact_score(query_vector: np.ndarray, row) -> float | None:
    vector = _decode_embedding_vector(row, query_vector.shape[0])
    if vector is None:
        return None
    query = query_vector.astype(np.float32, copy=False)
    query_norm = float(np.linalg.norm(query))
    if query_norm <= 0.0:
        return None
    return float(np.dot(query / query_norm, vector))


def _run_pipeline_search(
    runtime,
    image_bytes: bytes,
    k: int,
    *,
    is_live: bool,
    multi_face: bool,
) -> RawSearchOutcome:
    start = time.perf_counter()

    # --- Stage 1: detect + embed ---
    t_detect = time.perf_counter()
    detected_faces = (
        runtime.extractor.extract_embeddings(image_bytes)
        if multi_face
        else [runtime.extractor.extract_largest_embedding(image_bytes)]
    )
    t_embed_done = time.perf_counter()
    detect_embed_ms = (t_embed_done - t_detect) * 1000

    threshold = _threshold_for_pipeline(runtime.key, is_live=is_live)
    faces: list[RawFaceOutcome] = []
    search_modes: set[str] = set()
    candidate_ks: list[int] = []
    fallback_reasons: list[str] = []

    # --- Stage 2: search ---
    t_search = time.perf_counter()
    for face_index, detected in enumerate(detected_faces):
        matches, search_mode, candidate_k, fallback_reason = _search_index_dynamic(
            runtime,
            detected.embedding,
            k,
            threshold,
            is_live=is_live,
        )
        search_modes.add(search_mode)
        candidate_ks.append(candidate_k)
        if fallback_reason:
            fallback_reasons.append(fallback_reason)

        best_score = _best_score(matches)
        above = best_score is not None and best_score >= threshold
        faces.append(
            RawFaceOutcome(
                face_index=face_index,
                detection_score=detected.detection_score,
                face_bbox=detected.bbox,
                query_vector=detected.embedding,
                matches=matches,
                best_score=best_score,
                best_match_above_threshold=above,
                decision="match" if above else "unknown",
                search_mode=search_mode,
                candidate_k=candidate_k,
                fallback_reason=fallback_reason,
            )
        )
    t_search_done = time.perf_counter()
    faiss_ms = (t_search_done - t_search) * 1000

    best_score = max(
        (face.best_score for face in faces if face.best_score is not None),
        default=None,
    )
    any_match = any(face.best_match_above_threshold for face in faces)
    latency_ms = (time.perf_counter() - start) * 1000

    latency_breakdown = LatencyBreakdown(
        detect_ms=round(detect_embed_ms, 2),
        embed_ms=None,  # detect+embed are currently fused
        faiss_ms=round(faiss_ms, 2),
        search_ms=round(faiss_ms, 2),
        total_ms=round(latency_ms, 2),
    )
    search_mode = "mixed" if len(search_modes) > 1 else next(iter(search_modes), "none")

    return RawSearchOutcome(
        pipeline=runtime.key,
        model=runtime.extractor.model_name,
        faces=faces,
        latency_ms=latency_ms,
        threshold_used=threshold,
        best_score=best_score,
        best_match_above_threshold=any_match,
        decision="match" if any_match else "unknown",
        search_mode=search_mode,
        candidate_k=max(candidate_ks, default=0),
        fallback_reason=", ".join(sorted(set(fallback_reasons))) if fallback_reasons else None,
        multi_face_enabled=multi_face,
        latency_breakdown=latency_breakdown,
    )


def _hydrate_results(
    db: Session,
    pipeline: str,
    faces: list[RawFaceOutcome],
    threshold: float,
    limit_per_face: int,
) -> list[SearchResult]:
    repo = EmbeddingRepo(db)
    embedding_ids = [
        match.embedding_id
        for face in faces
        for match in face.matches
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
    multi_face = len(faces) > 1
    for face in faces:
        bbox = list(face.face_bbox) if face.face_bbox is not None else None
        buckets: dict[str, list[tuple[IndexMatch, object, float]]] = {}
        for match in face.matches:
            row = lookup.get(match.embedding_id)
            if row is None or row.person is None:
                continue
            # Skip deactivated embeddings or soft-deleted persons
            if not row.is_active or row.person.status != "active" or row.pipeline != pipeline:
                continue
            if _is_scale_label(row.person.label):
                continue
            score = _exact_score(face.query_vector, row)
            if score is None:
                continue
            buckets.setdefault(str(row.person.id), []).append((match, row, score))

        face_results: list[SearchResult] = []
        for person_matches in buckets.values():
            best_match, best_row, best_score = max(
                person_matches,
                key=lambda item: item[2],
            )
            support_floor = threshold
            if pipeline == "custom":
                support_floor = max(0.18, threshold - 0.06)
            support_count = sum(
                1 for _match, _row, score in person_matches if score >= support_floor
            )
            if not _is_supported_person_match(
                pipeline,
                best_score,
                support_count,
                threshold,
                best_row.person.label,
                multi_face=multi_face,
            ):
                continue

            person = best_row.person
            face_results.append(
                SearchResult(
                    pipeline=pipeline,
                    face_index=face.face_index,
                    detection_score=face.detection_score,
                    face_bbox=bbox,
                    person_id=str(person.id),
                    embedding_id=best_match.embedding_id,
                    score=best_score,
                    distance=1.0 - best_score,
                    label=person.label,
                    support_count=support_count,
                )
            )

        face_results.sort(key=lambda result: (result.score, result.support_count), reverse=True)
        if face_results:
            best = face_results[0]
            second = face_results[1] if len(face_results) > 1 else None
            if _is_ambiguous_person_ranking(pipeline, best, second, multi_face=multi_face):
                face_results = []
        results.extend(face_results[:limit_per_face])
    return _deduplicate_identity_across_faces(results)


def _response_from_outcome(
    db: Session,
    outcome: RawSearchOutcome,
    k: int,
    available_pipelines: list[str],
    upload_ms: float | None = None,
) -> SearchResponse:
    db_started = time.perf_counter()
    results = _hydrate_results(
        db,
        outcome.pipeline,
        outcome.faces,
        outcome.threshold_used,
        k,
    )
    db_ms = (time.perf_counter() - db_started) * 1000
    total_ms = (outcome.latency_ms or 0.0) + db_ms + (upload_ms or 0.0)
    latency_breakdown = outcome.latency_breakdown
    if latency_breakdown is not None:
        latency_breakdown = latency_breakdown.model_copy(
            update={
                "upload_ms": round(upload_ms, 2) if upload_ms is not None else None,
                "db_ms": round(db_ms, 2),
                "response_ms": round(db_ms, 2),
                "total_ms": round(total_ms, 2),
            }
        )
    matched_face_indices = {result.face_index for result in results}
    matched_faces = len(matched_face_indices)
    response_best_score = (
        max((result.score for result in results), default=None)
        if results
        else outcome.best_score
    )
    decision = "match" if matched_faces > 0 else "unknown"
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
        best_score=response_best_score,
        best_match_above_threshold=matched_faces > 0,
        decision=decision,
        pipeline=outcome.pipeline,
        latency_ms=total_ms,
        search_mode=outcome.search_mode,
        candidate_k=outcome.candidate_k,
        fallback_reason=outcome.fallback_reason,
        multi_face_enabled=outcome.multi_face_enabled,
        faces_processed=len(outcome.faces),
        available_pipelines=available_pipelines,
        detected_faces=detected_faces,
        latency_breakdown=latency_breakdown,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: Request,
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=100),
    pipeline: Literal["pretrained", "custom"] | None = Query(None),
    source: Literal["manual", "webcam"] | None = Query(None),
    multi_face: bool = Query(True),
    db: Session = Depends(get_db),
    registry: PipelineRegistry = Depends(get_pipeline_registry),
) -> SearchResponse:
    upload_started = time.perf_counter()
    try:
        image_bytes = await read_image_upload(
            file,
            max_bytes=settings.max_upload_bytes,
            allowed_types=allowed_content_types(settings.allowed_image_content_types),
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    upload_ms = (time.perf_counter() - upload_started) * 1000
    try:
        runtime = registry.resolve_search(pipeline)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        is_live = source == "webcam"
        outcome = _run_pipeline_search(
            runtime,
            image_bytes,
            1 if is_live else k,
            is_live=is_live,
            multi_face=multi_face,
        )
    except Exception as exc:  # noqa: BLE001
        raise _map_face_error(exc) from exc

    response = _response_from_outcome(
        db,
        outcome,
        1 if source == "webcam" else k,
        registry.available_pipelines(),
        upload_ms=upload_ms,
    )
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
