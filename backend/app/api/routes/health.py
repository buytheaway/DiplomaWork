"""Health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.config import settings

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    backend = settings.embedding_backend
    model = getattr(request.app.state, "extractor", None)
    model_name = model.model_name if model else "n/a"
    return {
        "status": "ok",
        "embedding_backend": backend,
        "model_name": model_name,
    }
