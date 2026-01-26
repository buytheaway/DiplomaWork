import logging
import random

import numpy as np
from fastapi import FastAPI

from app.api.routes import enroll, health, index, persons, search
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import SessionLocal
from app.services.embeddings.interface import DummyEmbeddingExtractor
from app.services.index.index_manager import IndexManager


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    random.seed(settings.seed)
    np.random.seed(settings.seed)

    app = FastAPI(title=settings.app_name)
    app.include_router(health.router, prefix=settings.api_v1_prefix)
    app.include_router(enroll.router, prefix=settings.api_v1_prefix)
    app.include_router(search.router, prefix=settings.api_v1_prefix)
    app.include_router(persons.router, prefix=settings.api_v1_prefix)
    app.include_router(index.router, prefix=settings.api_v1_prefix)

    @app.on_event("startup")
    def on_startup() -> None:
        if settings.testing:
            extractor = DummyEmbeddingExtractor()
        else:
            from app.services.embeddings.insightface_extractor import (
                InsightFaceEmbeddingExtractor,
            )

            extractor = InsightFaceEmbeddingExtractor(settings)
        app.state.extractor = extractor

        index_manager = IndexManager(settings)
        app.state.index_manager = index_manager

        if not settings.testing:
            try:
                with SessionLocal() as db:
                    loaded = index_manager.load_latest_snapshot(db)
                    logging.getLogger(__name__).info("Index loaded: %s", loaded)
            except Exception as exc:
                logging.getLogger(__name__).warning("Failed to load index snapshot: %s", exc)

    return app


app = create_app()
