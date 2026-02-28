"""FastAPI dependency‑injection providers."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.config import Settings, settings
from app.db.session import SessionLocal
from app.services.embeddings.interface import EmbeddingExtractor
from app.services.index.index_manager import IndexManager


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
