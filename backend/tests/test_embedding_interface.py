"""Unit tests for embedding extractor interface helpers."""

from __future__ import annotations

import os

os.environ.setdefault("TESTING", "true")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from app.services.embeddings.interface import DummyEmbeddingExtractor


def test_dummy_extract_embeddings_wraps_single_face():
    ext = DummyEmbeddingExtractor(dim=8)

    faces = ext.extract_embeddings(b"image-bytes")

    assert len(faces) == 1
    assert faces[0].embedding.shape == (8,)
    assert faces[0].detection_score is None
    assert faces[0].bbox is None
