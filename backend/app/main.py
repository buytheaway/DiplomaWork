"""FastAPI application factory.

* Wires routes, creates the embedding extractor (plugin), and boots the index.
* Heavy ML libraries are loaded **only** when ``EMBEDDING_BACKEND`` is not ``dummy``.
"""

from __future__ import annotations

import logging
import random
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import numpy as np
from fastapi import FastAPI

from app.api.routes import enroll, health, index, persons, search
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.embeddings.interface import create_extractor
from app.services.index.index_manager import IndexManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook (replaces deprecated ``on_event``)."""
    settings = get_settings()

    # ── extractor (plugin) ───────────────────────────────────────────
    app.state.extractor = create_extractor(settings)
    logger = logging.getLogger(__name__)
    logger.info(
        "Embedding extractor ready: backend=%s model=%s dim=%d",
        settings.embedding_backend,
        app.state.extractor.model_name,
        app.state.extractor.dim,
    )

    # ── vector index ─────────────────────────────────────────────────
    index_manager = IndexManager(settings)
    app.state.index_manager = index_manager

    if not settings.testing:
        try:
            from app.db.session import SessionLocal

            with SessionLocal() as db:
                loaded = index_manager.load_latest_snapshot(db)
                logger.info("Index loaded from snapshot: %s", loaded)
        except Exception as exc:
            logger.warning("Failed to load index snapshot: %s", exc)

    yield  # ── application is running ──

    # shutdown (nothing required for now)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    random.seed(settings.seed)
    np.random.seed(settings.seed)

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(health.router, prefix=settings.api_v1_prefix)
    app.include_router(enroll.router, prefix=settings.api_v1_prefix)
    app.include_router(search.router, prefix=settings.api_v1_prefix)
    app.include_router(persons.router, prefix=settings.api_v1_prefix)
    app.include_router(index.router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
