"""Face enrolment endpoint."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_pipeline_registry
from app.api.schemas.enroll import EnrollmentItem, EnrollResponse
from app.core.config import settings
from app.services.embeddings.interface import (
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.services.runtime.pipeline_registry import PipelineRegistry
from app.services.storage.audit import record_audit_event
from app.services.storage.repositories import EmbeddingRepo, PersonRepo
from app.services.storage.uploads import (
    UploadValidationError,
    allowed_content_types,
    read_image_upload,
)

router = APIRouter()


def _raise_face_error(exc: Exception) -> None:
    if isinstance(exc, UploadValidationError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, InvalidImageError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, (NoFaceDetectedError, MultipleFacesDetectedError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail="Embedding extraction failed") from exc


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(
    request: Request,
    file: UploadFile = File(...),
    label: str | None = Form(None),
    pipeline: Literal["pretrained", "custom", "both"] | None = Form(None),
    db: Session = Depends(get_db),
    registry: PipelineRegistry = Depends(get_pipeline_registry),
) -> EnrollResponse:
    try:
        image_bytes = await read_image_upload(
            file,
            max_bytes=settings.max_upload_bytes,
            allowed_types=allowed_content_types(settings.allowed_image_content_types),
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        runtimes = registry.resolve_enroll(pipeline)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    computed: list[tuple[str, str, object]] = []
    for runtime in runtimes:
        try:
            embedding = runtime.extractor.extract_embedding(image_bytes)
        except Exception as exc:  # noqa: BLE001
            _raise_face_error(exc)
        computed.append((runtime.key, runtime.extractor.model_name, embedding))

    person_repo = PersonRepo(db)
    embedding_repo = EmbeddingRepo(db)

    person = person_repo.create(label=label)
    db.flush()

    saved_rows: list[tuple[str, str, object, object]] = []
    for runtime_key, model_name, embedding in computed:
        embedding_row = embedding_repo.create(
            person_id=person.id,
            pipeline=runtime_key,
            model=model_name,
            dim=int(embedding.shape[0]),
            vector=embedding.tobytes(),
        )
        db.flush()
        saved_rows.append((runtime_key, model_name, embedding, embedding_row))

    db.commit()
    db.refresh(person)
    for _, _, _, embedding_row in saved_rows:
        db.refresh(embedding_row)

    try:
        touched_pipelines: set[str] = set()
        for runtime_key, _model_name, embedding, embedding_row in saved_rows:
            runtime = registry.get(runtime_key)  # type: ignore[arg-type]
            runtime.index_manager.add_embedding(str(embedding_row.id), embedding)
            touched_pipelines.add(runtime_key)

        if settings.auto_save_index:
            for runtime_key in touched_pipelines:
                runtime = registry.get(runtime_key)  # type: ignore[arg-type]
                runtime.index_manager.save_snapshot(db)
            db.commit()
    except Exception as exc:  # pragma: no cover - exercised in real runtime
        for _, _, _, embedding_row in saved_rows:
            embedding_repo.deactivate(embedding_row.id)
        db.commit()
        raise HTTPException(status_code=500, detail="Index update failed") from exc

    enrollments = [
        EnrollmentItem(
            pipeline=runtime_key,
            embedding_id=str(embedding_row.id),
            model=model_name,
            dim=int(embedding.shape[0]),
        )
        for runtime_key, model_name, embedding, embedding_row in saved_rows
    ]
    first = enrollments[0]

    response = EnrollResponse(
        person_id=str(person.id),
        embedding_id=first.embedding_id,
        faces_detected=1,
        model=first.model,
        dim=first.dim,
        pipeline=pipeline or registry.default_pipeline,
        available_pipelines=registry.available_pipelines(),
        enrollments=enrollments,
    )
    record_audit_event(
        db,
        request,
        event_type="enroll",
        status_code=200,
        details={
            "person_id": str(person.id),
            "pipelines": [item.pipeline for item in enrollments],
            "faces_detected": 1,
        },
    )
    return response
