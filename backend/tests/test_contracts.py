import os
import uuid

import numpy as np

os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

from app.api.schemas.enroll import EnrollResponse
from app.core.config import get_settings
from app.services.index.faiss_index import FaissIndex
from app.services.index.index_manager import IndexManager


def test_enroll_schema_validation():
    payload = {
        "person_id": str(uuid.uuid4()),
        "embedding_id": str(uuid.uuid4()),
        "faces_detected": 1,
        "model": "test",
        "dim": 512,
    }
    EnrollResponse(**payload)


def test_faiss_index_add_search():
    index = FaissIndex(dim=3, index_type="flat", params={}, seed=42)
    index.add_embedding("a", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    index.add_embedding("b", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert results
    assert results[0].embedding_id == "a"


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
