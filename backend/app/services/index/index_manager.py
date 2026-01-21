from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.services.index.faiss_index import FaissIndex
from app.services.storage.repositories import EmbeddingRepo, IndexSnapshotRepo


@dataclass(frozen=True)
class IndexMatch:
    embedding_id: str
    score: float
    distance: float


class IndexManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index_path = Path(settings.index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index = self._create_index(
            settings.index_type, self.default_params_for(settings.index_type)
        )
        self._index_type = settings.index_type
        self._params = self.default_params_for(settings.index_type)

    def default_params_for(self, index_type: str) -> dict[str, Any]:
        if index_type == "hnsw":
            return {
                "m": self.settings.hnsw_m,
                "ef_construction": self.settings.hnsw_ef_construction,
                "ef_search": self.settings.hnsw_ef_search,
            }
        if index_type == "ivfpq":
            return {
                "nlist": self.settings.ivfpq_nlist,
                "m": self.settings.ivfpq_m,
                "nbits": self.settings.ivfpq_nbits,
            }
        return {}

    def _create_index(self, index_type: str, params: dict[str, Any]) -> FaissIndex:
        return FaissIndex(dim=512, index_type=index_type, params=params, seed=self.settings.seed)

    def load_latest_snapshot(self, db: Session) -> bool:
        repo = IndexSnapshotRepo(db)
        snapshot = repo.get_latest()
        if snapshot and Path(snapshot.path).exists():
            self._index = self._create_index(snapshot.index_type, snapshot.params)
            self._index.load(snapshot.path)
            self._index_type = snapshot.index_type
            self._params = snapshot.params
            return True
        return False

    def add_embedding(self, embedding_id: str, vector: np.ndarray) -> int:
        return self._index.add_embedding(embedding_id, vector)

    def search(self, vector: np.ndarray, k: int) -> list[IndexMatch]:
        results = self._index.search(vector, k)
        return [IndexMatch(r.embedding_id, r.score, r.distance) for r in results]

    def save_snapshot(self, db: Session) -> None:
        self._index.save(str(self.index_path))
        repo = IndexSnapshotRepo(db)
        repo.create(
            index_type=self._index_type,
            params=self._params,
            path=str(self.index_path),
            embeddings_count=self._index.count(),
        )

    def rebuild(self, db: Session, index_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if not params:
            params = self.default_params_for(index_type)
        embeddings = EmbeddingRepo(db).get_active_embeddings()
        self._index = self._create_index(index_type, params)
        self._index_type = index_type
        self._params = params

        if embeddings:
            vectors = np.vstack([np.frombuffer(e.vector, dtype=np.float32) for e in embeddings])
            self._index.train(vectors)
            for embedding, vector in zip(embeddings, vectors):
                self._index.add_embedding(str(embedding.id), vector)

        self.save_snapshot(db)
        return self.stats()

    def stats(self) -> dict[str, Any]:
        stats = self._index.stats()
        stats["loaded"] = self._index.count() > 0
        return stats

    def count(self) -> int:
        return self._index.count()
