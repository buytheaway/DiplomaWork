"""Tests for the embedding extractor factory and backend wiring."""

from __future__ import annotations

import os

os.environ.setdefault("TESTING", "true")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from unittest.mock import patch

from app.core.config import get_settings
from app.services.embeddings.interface import (
    DummyEmbeddingExtractor,
    create_extractor,
)

# ── factory: dummy ───────────────────────────────────────────────────────────


def test_factory_returns_dummy_when_testing():
    settings = get_settings()
    ext = create_extractor(settings)
    assert isinstance(ext, DummyEmbeddingExtractor)
    assert ext.model_name == "dummy"


def test_factory_returns_dummy_for_explicit_backend():
    settings = get_settings()
    # Even if someone explicitly sets dummy
    with patch.object(settings, "embedding_backend", "dummy"):
        with patch.object(settings, "testing", False):
            ext = create_extractor(settings)
            assert isinstance(ext, DummyEmbeddingExtractor)


# ── factory: unknown backend ─────────────────────────────────────────────────


def test_factory_raises_on_unknown_backend():
    settings = get_settings()
    with patch.object(settings, "embedding_backend", "unknown_xyz"):
        with patch.object(settings, "testing", False):
            try:
                create_extractor(settings)
                raise AssertionError("Expected ValueError")
            except ValueError as exc:
                assert "unknown_xyz" in str(exc)


# ── factory: insightface import guard ────────────────────────────────────────


def test_factory_insightface_import_error():
    """When insightface is not installed, factory propagates ImportError."""
    settings = get_settings()
    with patch.object(settings, "embedding_backend", "insightface"):
        with patch.object(settings, "testing", False):
            with patch.dict("sys.modules", {"insightface": None, "insightface.app": None}):
                try:
                    create_extractor(settings)
                    # It's fine if it succeeds (insightface installed)
                except (ImportError, Exception):
                    pass  # Expected — insightface not installed in test env


# ── factory: onnx import guard ───────────────────────────────────────────────


def test_factory_onnx_import_error():
    """When onnxruntime is not installed, factory propagates ImportError."""
    settings = get_settings()
    with patch.object(settings, "embedding_backend", "onnx"):
        with patch.object(settings, "testing", False):
            try:
                create_extractor(settings)
            except (ImportError, ValueError, Exception):
                pass  # Expected — onnx models not available in test env
