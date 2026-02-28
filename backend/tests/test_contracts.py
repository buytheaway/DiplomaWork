"""Contract / unit tests — schemas, FAISS index, index manager, dummy extractor."""

import os
import uuid

import numpy as np

os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

from app.api.schemas.enroll import EnrollResponse
from app.core.config import get_settings
from app.services.embeddings.interface import (
    DummyEmbeddingExtractor,
    InvalidImageError,
    create_extractor,
)
from app.services.index.faiss_index import FaissIndex
from app.services.index.index_manager import IndexManager


# ── schema validation ────────────────────────────────────────────────────────


def test_enroll_schema_validation():
    payload = {
        "person_id": str(uuid.uuid4()),
        "embedding_id": str(uuid.uuid4()),
        "faces_detected": 1,
        "model": "test",
        "dim": 512,
    }
    resp = EnrollResponse(**payload)
    assert resp.dim == 512


# ── FAISS index ──────────────────────────────────────────────────────────────


def test_faiss_index_add_search():
    index = FaissIndex(dim=3, index_type="flat", params={}, seed=42)
    index.add_embedding("a", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    index.add_embedding("b", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert results
    assert results[0].embedding_id == "a"


def test_faiss_index_deterministic():
    """Same data + same seed → same search results."""
    for _ in range(3):
        idx = FaissIndex(dim=4, index_type="flat", params={}, seed=42)
        idx.add_embedding("x", np.array([1, 0, 0, 0], dtype=np.float32))
        idx.add_embedding("y", np.array([0, 1, 0, 0], dtype=np.float32))
        res = idx.search(np.array([0.9, 0.1, 0, 0], dtype=np.float32), k=2)
        assert res[0].embedding_id == "x"
        assert res[1].embedding_id == "y"


def test_faiss_index_empty_search():
    idx = FaissIndex(dim=3, index_type="flat", params={}, seed=42)
    assert idx.search(np.array([1, 0, 0], dtype=np.float32), k=5) == []


# ── index manager ────────────────────────────────────────────────────────────


def test_index_manager_stats_keys():
    settings = get_settings()
    manager = IndexManager(settings)
    stats = manager.stats()
    expected_keys = {
        "index_type",
        "params",
        "embeddings_count",
        "memory_estimate",
        "memory_estimate_bytes",
        "loaded",
        "is_trained",
        "file_path",
        "last_snapshot_id",
    }
    assert expected_keys.issubset(stats.keys())
    assert stats["loaded"] is False


def test_index_manager_uses_embedding_dim():
    settings = get_settings()
    manager = IndexManager(settings)
    assert manager.dim == settings.embedding_dim


# ── dummy extractor ──────────────────────────────────────────────────────────


def test_dummy_extractor_returns_unit_vector():
    ext = DummyEmbeddingExtractor(dim=128)
    vec = ext.extract_embedding(b"\x00\x01\x02")
    assert vec.shape == (128,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_dummy_extractor_rejects_empty_bytes():
    ext = DummyEmbeddingExtractor()
    try:
        ext.extract_embedding(b"")
        assert False, "Expected InvalidImageError"
    except InvalidImageError:
        pass


def test_create_extractor_factory_dummy():
    settings = get_settings()
    ext = create_extractor(settings)
    assert ext.model_name == "dummy"
    assert ext.dim == settings.embedding_dim
