"""FastAPI application factory.

* Wires routes, creates the embedding runtime pipelines, and boots the indices.
* Heavy ML libraries are loaded lazily per configured pipeline.
"""

from __future__ import annotations

import logging
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.responses import JSONResponse

from app.api.routes import enroll, health, index, persons, search
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.runtime.pipeline_registry import PipelineRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook (replaces deprecated ``on_event``)."""
    settings = get_settings()
    logger = logging.getLogger(__name__)

    registry = PipelineRegistry(settings)
    registry.initialize()
    app.state.pipeline_registry = registry
    app.state.extractor = registry.default_runtime.extractor
    app.state.index_manager = registry.default_runtime.index_manager

    logger.info(
        "Default pipeline ready: key=%s backend=%s model=%s dim=%d available=%s",
        registry.default_pipeline,
        registry.default_runtime.backend,
        registry.default_runtime.extractor.model_name,
        registry.default_runtime.extractor.dim,
        registry.available_pipelines(),
    )

    if not settings.testing:
        try:
            from app.db.session import SessionLocal

            with SessionLocal() as db:
                loaded = registry.load_latest_snapshots(db)
                logger.info("Index snapshots loaded: %s", loaded)
        except Exception as exc:  # pragma: no cover - exercised in real runtime
            logger.warning("Failed to load index snapshots: %s", exc)

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    random.seed(settings.seed)

    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.api_key:
        _api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

        @app.middleware("http")
        async def _check_api_key(request: Request, call_next):
            path = request.url.path
            if path.endswith("/health") or path.startswith("/docs") or path.startswith("/openapi"):
                return await call_next(request)
            key = request.headers.get("X-API-Key", "")
            if key != settings.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
            return await call_next(request)

    app.include_router(health.router, prefix=settings.api_v1_prefix)
    app.include_router(enroll.router, prefix=settings.api_v1_prefix)
    app.include_router(search.router, prefix=settings.api_v1_prefix)
    app.include_router(persons.router, prefix=settings.api_v1_prefix)
    app.include_router(index.router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
