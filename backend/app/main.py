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
from starlette.responses import JSONResponse

from app.api.routes import enroll, health, index, persons, search
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.security.auth import classify_api_key
from app.security.rate_limit import (
    InMemoryRateLimiter,
    rate_limit_identity,
    rate_limit_rule_for_request,
)
from app.services.runtime.pipeline_registry import PipelineRegistry
from app.services.storage.repositories import AuditLogRepo


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
                pruned = AuditLogRepo(db).prune_older_than(settings.audit_retention_days)
                if pruned:
                    db.commit()
                    logger.info("Pruned %d expired audit log entries", pruned)
        except Exception as exc:  # pragma: no cover - exercised in real runtime
            logger.warning("Failed to load index snapshots: %s", exc)

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    random.seed(settings.seed)

    if not settings.testing:
        missing: list[str] = []
        if not settings.api_key:
            missing.append("API_KEY")
        if not settings.admin_api_key:
            missing.append("ADMIN_API_KEY")
        if not settings.data_encryption_key:
            missing.append("DATA_ENCRYPTION_KEY")
        if not settings.snapshot_encryption_key:
            missing.append("SNAPSHOT_ENCRYPTION_KEY")
        if missing:
            raise RuntimeError(
                "Secure backend configuration is incomplete. Missing: "
                + ", ".join(missing)
            )
        if settings.api_key == settings.admin_api_key:
            raise RuntimeError("API_KEY and ADMIN_API_KEY must be different")

    docs_enabled = not bool(settings.api_key and not settings.testing)
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.rate_limit_enabled:
        rate_limiter = InMemoryRateLimiter()

        @app.middleware("http")
        async def _rate_limit(request: Request, call_next):
            rule = rate_limit_rule_for_request(
                settings,
                request.method,
                request.url.path,
            )
            if rule is not None:
                identity = rate_limit_identity(settings, request)
                decision = rate_limiter.check(
                    identity=identity,
                    category=rule.category,
                    limit=rule.limit,
                )
                if not decision.allowed:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded. Try again later."},
                        headers={"Retry-After": str(decision.retry_after_seconds)},
                    )
            return await call_next(request)

    if settings.api_key and not settings.testing:
        @app.middleware("http")
        async def _check_api_key(request: Request, call_next):
            path = request.url.path
            if path.endswith("/health"):
                return await call_next(request)
            key = request.headers.get("X-API-Key", "")
            role = classify_api_key(settings, key)
            if role is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
            request.state.actor_role = role
            return await call_next(request)

    app.include_router(health.router, prefix=settings.api_v1_prefix)
    app.include_router(enroll.router, prefix=settings.api_v1_prefix)
    app.include_router(search.router, prefix=settings.api_v1_prefix)
    app.include_router(persons.router, prefix=settings.api_v1_prefix)
    app.include_router(index.router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
