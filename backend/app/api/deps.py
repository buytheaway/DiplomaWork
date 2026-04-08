"""FastAPI dependency‑injection providers."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, settings
from app.db.session import SessionLocal
from app.services.embeddings.interface import EmbeddingExtractor
from app.services.index.index_manager import IndexManager
from app.services.runtime.pipeline_registry import PipelineRegistry


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_settings() -> Settings:
    return settings


def get_extractor(request: Request) -> EmbeddingExtractor:
    return request.app.state.extractor


def get_index_manager(request: Request) -> IndexManager:
    return request.app.state.index_manager


def get_pipeline_registry(request: Request) -> PipelineRegistry:
    return request.app.state.pipeline_registry


def get_actor_role(request: Request) -> str:
    role = getattr(request.state, "actor_role", None)
    if role is None and settings.testing:
        return "admin"
    if role is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return role


def require_admin(request: Request) -> str:
    role = get_actor_role(request)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin API key required for this action")
    return role
