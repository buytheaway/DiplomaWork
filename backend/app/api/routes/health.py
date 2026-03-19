"""Health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    registry = request.app.state.pipeline_registry
    default_runtime = registry.default_runtime
    payload = registry.status()
    payload.update(
        {
            "status": "ok",
            "embedding_backend": default_runtime.backend,
            "model_name": default_runtime.extractor.model_name,
            "strict_single_face_enroll": True,
            "multi_face_search": True,
        }
    )
    return payload
